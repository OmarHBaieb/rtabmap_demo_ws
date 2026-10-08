"""
Capture N paires RGB/depth synchronisées depuis Gazebo, à intervalles réguliers,
pendant que le robot est téléopéré ou se déplace (via teleop/nav manuel) pour
varier les points de vue.
"""
import rclpy, time, os
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

OUTPUT_DIR = "gazebo_frames"
N_FRAMES = 80
INTERVAL_SEC = 1.5  # laisse le temps de bouger le robot entre 2 captures

class BatchGrab(Node):
    def __init__(self):
        super().__init__('batch_grab')
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.create_subscription(Image, '/camera/image_raw', self.cb_rgb, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.cb_depth, 10)

    def cb_rgb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def cb_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')


os.makedirs(OUTPUT_DIR, exist_ok=True)
rclpy.init()
node = BatchGrab()

# attendre le premier message des deux topics
while node.rgb is None or node.depth is None:
    rclpy.spin_once(node, timeout_sec=0.5)

print(f"Capture de {N_FRAMES} frames, une toutes les {INTERVAL_SEC}s.")
print("Déplace/tourne le robot manuellement entre les captures (téléop dans un autre terminal) pour varier les points de vue.")

for i in range(N_FRAMES):
    rclpy.spin_once(node, timeout_sec=0.5)
    name = f"{i:04d}"
    np.save(f"{OUTPUT_DIR}/{name}_rgb.npy", node.rgb)
    np.save(f"{OUTPUT_DIR}/{name}_depth.npy", node.depth)
    print(f"[{i+1}/{N_FRAMES}] capturé -> {name}")
    time.sleep(INTERVAL_SEC)

print("Terminé.")