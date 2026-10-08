#!/usr/bin/env python3
"""
detector_node.py

Nœud ROS 2 remplaçant find_object_2d dans rtabmap_demo_ws.

S'abonne à RGB + depth + camera_info synchronisés, fait l'inférence avec
multimodal_transformer.pt (segmentation 3 classes : sol/obstacle/inconnu),
publie un masque de segmentation et un overlay visuel.

Topics d'entrée par défaut (identiques aux remappings de find_object_demo.launch.py,
    rgb/image_rect_color            -> /camera/data_throttled_image
    depth_registered/image_raw      -> /camera/data_throttled_image_depth
    depth_registered/camera_info    -> /camera/data_throttled_camera_info

Topics de sortie :
    segmentation/mask     (sensor_msgs/Image, mono8, valeurs 0/1/2)
    segmentation/overlay  (sensor_msgs/Image, bgr8, RGB + masque coloré superposé)
"""

import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


IMG_SIZE = 224
DEPTH_MAX_M = 3.0  # clip depth à 3 m, cohérent avec l'entraînement prévu

# Normalisation ImageNet pour la branche RGB (attendue par DINOv2)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Couleurs BGR pour l'overlay : 0=sol, 1=obstacle, 2=inconnu
CLASS_COLORS = np.array([
    [0, 200, 0],     # sol -> vert
    [0, 0, 220],     # obstacle -> rouge
    [200, 200, 0],   # inconnu -> cyan/jaune
], dtype=np.uint8)


def numpy_to_imgmsg(img: np.ndarray, encoding: str, header) -> Image:
    img = np.ascontiguousarray(img)
    msg = Image()
    msg.header = header
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(img.strides[0])
    msg.data = img.tobytes()
    return msg


class DetectorNode(Node):
    def __init__(self):
        super().__init__('transformer_object_detector')

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('rgb_topic', 'rgb/image_rect_color')
        self.declare_parameter('depth_topic', 'depth_registered/image_raw')
        self.declare_parameter('camera_info_topic', 'depth_registered/camera_info')
        self.declare_parameter('sync_slop', 0.05)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        device_str = self.get_parameter('device').get_parameter_value().string_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value

        if not model_path:
            self.get_logger().error(
                "Paramètre 'model_path' vide. Lancer avec: "
                "--ros-args -p model_path:=/chemin/vers/multimodal_transformer.pt")
            raise RuntimeError("model_path manquant")

        self.device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
        self.get_logger().info(f"Chargement du modèle TorchScript : {model_path} (device={self.device})")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.get_logger().info("Modèle chargé.")

        self.bridge = CvBridge()

        self.mask_pub = self.create_publisher(Image, 'segmentation/mask', 10)
        self.overlay_pub = self.create_publisher(Image, 'segmentation/overlay', 10)

        rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=sync_slop)
        self.ts.registerCallback(self.on_synced_msgs)

        self.get_logger().info(
            f"Abonné à: rgb='{rgb_topic}', depth='{depth_topic}', info='{info_topic}'")

    def preprocess_rgb(self, rgb_img: np.ndarray) -> torch.Tensor:
        # rgb_img: HxWx3 BGR uint8 (cv_bridge par défaut)
        rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        return tensor

    def preprocess_depth(self, depth_img: np.ndarray) -> torch.Tensor:
        # depth_img: HxW, float32 en mètres (convention ROS 32FC1) OU uint16 en mm (16UC1)
        is_uint_mm = depth_img.dtype != np.float32
        depth = depth_img.astype(np.float32)
        if is_uint_mm:
            depth = depth / 1000.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=DEPTH_MAX_M, neginf=0.0)
        # Gazebo renvoie une valeur sentinelle (~300m) pour le "hors-portée"
        # (ciel, rayons qui ne touchent rien) — ce n'est pas une vraie mesure
        # de profondeur. Sans ce filtre, ces pixels dominent la distribution
        # après clip et écrasent le signal réel (bug root-caused: le clip à
        # DEPTH_MAX_M faisait que la quasi-totalité de l'image valait 1.0).
        depth[depth > 50.0] = DEPTH_MAX_M
        depth = np.clip(depth, 0.0, DEPTH_MAX_M)
        depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        depth = depth / DEPTH_MAX_M  # normalisation [0, 1]
        tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]
        return tensor

    def on_synced_msgs(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Échec conversion image: {e}")
            return

        rgb_tensor = self.preprocess_rgb(rgb_cv)
        depth_tensor = self.preprocess_depth(depth_cv)

        # NOTE: signature exacte à valider une fois le modèle réellement chargé :
        # print(self.model.code) au démarrage pour confirmer l'ordre des arguments.
        with torch.no_grad():
            logits = self.model(rgb_tensor, depth_tensor)  # [1, 3, 224, 224]
            # Le canal 2 (inconnu) n'est jamais entraîné (ignore_index=255 côté
            # entraînement) — on restreint l'argmax aux 2 canaux réellement
            # appris pour ne jamais produire "inconnu" à partir d'un canal mort.
            pred_mask = torch.argmax(logits[:, :2], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Publication du masque brut (0/1/2)
        mask_msg = numpy_to_imgmsg(pred_mask, 'mono8', rgb_msg.header)
        mask_msg.header = rgb_msg.header
        self.mask_pub.publish(mask_msg)

        # Publication de l'overlay coloré
        color_mask = CLASS_COLORS[pred_mask]  # [224,224,3] BGR
        rgb_resized = cv2.resize(rgb_cv, (IMG_SIZE, IMG_SIZE))
        overlay = cv2.addWeighted(rgb_resized, 0.6, color_mask, 0.4, 0)
        overlay_msg = numpy_to_imgmsg(overlay, 'bgr8', rgb_msg.header)
        overlay_msg.header = rgb_msg.header
        self.overlay_pub.publish(overlay_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
