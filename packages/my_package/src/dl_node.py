#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO  # Ensure YOLO library is installed


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # Static parameters
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_vehicle')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # Bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # Create window for visualization
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # Load YOLO model
        self.yolo_model = YOLO('/path/to/best.pt')  # Replace with the correct path to your YOLO weights
        # Construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        # Convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Process the image with YOLO
        results = self.yolo_model.predict(source=image, save=False, save_txt=False, conf=0.5)  # Adjust confidence threshold

        # Draw detections on the image
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]
                conf = bbox.conf[0]
                cls = int(bbox.cls[0])
                label = f"{self.yolo_model.names[cls]} {conf:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow(self._window, image)
        cv2.waitKey(1)


if __name__ == '__main__':
    # Create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # Keep spinning
    rospy.spin()
