import rclpy, torch, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Grab(Node):
    def __init__(self):
        super().__init__('grab')
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.create_subscription(Image, '/camera/image_raw', self.cb_rgb, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.cb_depth, 10)

    def cb_rgb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def cb_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')

rclpy.init()
node = Grab()
while node.rgb is None or node.depth is None:
    rclpy.spin_once(node, timeout_sec=0.5)

np.save('rgb_frame.npy', node.rgb)
np.save('depth_frame.npy', node.depth)
print("depth stats:", np.nanmin(node.depth), np.nanmax(node.depth), np.nanmean(node.depth))