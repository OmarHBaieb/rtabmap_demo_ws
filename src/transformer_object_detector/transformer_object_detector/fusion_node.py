#!/usr/bin/env python3
"""
fusion_node.py

Fusionne RGB + depth + masque de segmentation (/segmentation/mask) en :
- un PointCloud2 annoté complet (rgb couleur de classe + label) : /segmentation/pointcloud
- deux PointCloud2 simples (xyz uniquement) pour Nav2 costmap :
    /segmentation/points_ground     (points classés "sol")
    /segmentation/points_obstacles  (points classés "obstacle")
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import message_filters

CLASS_COLORS = np.array([
    [0, 200, 0],     # sol -> vert
    [0, 0, 220],     # obstacle -> rouge
    [200, 200, 0],   # inconnu -> cyan/jaune
], dtype=np.uint8)


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.declare_parameter('rgb_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('mask_topic', '/segmentation/mask')
        self.declare_parameter('sync_slop', 0.3)
        self.declare_parameter('stride', 2)
        self.declare_parameter('depth_max_m', 5.0)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        mask_topic = self.get_parameter('mask_topic').value
        sync_slop = self.get_parameter('sync_slop').value
        self.stride = max(1, int(self.get_parameter('stride').value))
        self.depth_max = float(self.get_parameter('depth_max_m').value)

        self.bridge = CvBridge()
        self.cloud_pub = self.create_publisher(PointCloud2, 'segmentation/pointcloud', 5)
        self.obstacle_pub = self.create_publisher(PointCloud2, 'segmentation/points_obstacles', 5)
        self.ground_pub = self.create_publisher(PointCloud2, 'segmentation/points_ground', 5)

        rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        mask_sub = message_filters.Subscriber(self, Image, mask_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub, mask_sub], queue_size=10, slop=sync_slop)
        self.ts.registerCallback(self.on_synced)

        self.get_logger().info(
            f"Fusion Node abonné à: rgb={rgb_topic}, depth={depth_topic}, "
            f"info={info_topic}, mask={mask_topic}")

    def on_synced(self, rgb_msg, depth_msg, info_msg, mask_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f"Échec conversion: {e}")
            return

        h, w = depth.shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        is_uint_mm = depth.dtype != np.float32
        depth_m = depth.astype(np.float32)
        if is_uint_mm:
            depth_m = depth_m / 1000.0  # mm -> m

        fx, fy = info_msg.k[0], info_msg.k[4]
        cx, cy = info_msg.k[2], info_msg.k[5]

        s = self.stride
        us, vs = np.meshgrid(np.arange(0, w, s), np.arange(0, h, s))
        z = depth_m[vs, us]

        valid = np.isfinite(z) & (z > 0.05) & (z < self.depth_max)
        us, vs, z = us[valid], vs[valid], z[valid]

        if z.size == 0:
            return

        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        labels = mask[vs, us].astype(np.uint32)
        colors = CLASS_COLORS[mask[vs, us]]  # couleur de CLASSE (pas la couleur réelle)

        r = colors[:, 2].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 0].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b
        rgb_float = rgb_packed.view(np.float32)

        points = np.zeros(x.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('rgb', np.float32), ('label', np.uint32),
        ])
        points['x'] = x
        points['y'] = y
        points['z'] = z
        points['rgb'] = rgb_float
        points['label'] = labels

        self.cloud_pub.publish(self.array_to_pointcloud2(points, depth_msg.header))

        # --- Nuages séparés pour Nav2 costmap (xyz uniquement) ---
        xyz_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]

        obstacle_mask = (labels == 1)
        if obstacle_mask.any():
            obs_pts = np.zeros(int(obstacle_mask.sum()), dtype=xyz_dtype)
            obs_pts['x'] = x[obstacle_mask]
            obs_pts['y'] = y[obstacle_mask]
            obs_pts['z'] = z[obstacle_mask]
            self.obstacle_pub.publish(self.array_to_pointcloud2_xyz(obs_pts, depth_msg.header))

        ground_mask = (labels == 0)
        if ground_mask.any():
            grd_pts = np.zeros(int(ground_mask.sum()), dtype=xyz_dtype)
            grd_pts['x'] = x[ground_mask]
            grd_pts['y'] = y[ground_mask]
            grd_pts['z'] = z[ground_mask]
            self.ground_pub.publish(self.array_to_pointcloud2_xyz(grd_pts, depth_msg.header))

    def array_to_pointcloud2(self, points, header):
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_bigendian = False
        msg.is_dense = False
        msg.point_step = points.dtype.itemsize
        msg.row_step = msg.point_step * points.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),
        ]
        msg.data = points.tobytes()
        return msg

    def array_to_pointcloud2_xyz(self, points, header):
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_bigendian = False
        msg.is_dense = False
        msg.point_step = points.dtype.itemsize
        msg.row_step = msg.point_step * points.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = points.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
