import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import json

class RVizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')
        self.subscription = self.create_subscription(
            PointStamped,
            '/ball/kf_pos',
            self.listener_callback,
            100)
        self.marker_pub = self.create_publisher(MarkerArray, '/ball/trajectory_markers', 10)
        self.trajectories = {}  # ball_id -> list of (x, y, z)
        self.marker_id = 0

    def listener_callback(self, msg):
        ball_id_str, _ = msg.header.frame_id.split('_')
        ball_id = int(ball_id_str)
        x, y, z = msg.point.x, msg.point.y, msg.point.z

        if ball_id not in self.trajectories:
            self.trajectories[ball_id] = []
        self.trajectories[ball_id].append((x, y, z))

        # Publish updated markers
        marker_array = MarkerArray()
        for bid, points in self.trajectories.items():
            marker = Marker()
            marker.header.frame_id = 'world'  # Assuming world frame; adjust if needed
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'trajectory'
            marker.id = bid
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01  # Line width
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) if bid == 0 else ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = RVizVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()