#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO  # Ensure YOLO library is installed
import time  # For FPS measurement

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        # Static parameters
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_vehicle')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # Create a window for visualization
        #self._window = "YOLO Processed Image"
        #cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        # Load YOLO model
        model_path = os.path.join(os.getcwd(), 'best.pt')  # Adjust path as needed
        if not os.path.exists(model_path):
            rospy.logerr(f"YOLO model not found at {model_path}. Exiting...")
            exit(1)

        self.yolo_model = YOLO(model_path)

        # Construct ROS subscribers and publishers
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.image_pub = rospy.Publisher("/yolo_processed_image", CompressedImage, queue_size=1)

        rospy.loginfo("✅ CameraReaderNode initialized successfully!")



    def callback(self, msg):
        start_time = time.time()  # Start FPS timer

        # Convert JPEG bytes to OpenCV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Ensure image is valid
        if image is None or image.size == 0:
            rospy.logwarn("⚠️ Received an empty image! Skipping frame...")
            return  

        # Run YOLO on the image
        results = self.yolo_model.predict(source=image, save=False, save_txt=False, conf=0.25, imgsz=320)  # Faster inference

        # Check if objects are detected
        if not results or not results[0].boxes:
            rospy.logwarn("⚠️ YOLO did not detect any objects in this frame.")

        # Draw bounding boxes on the image
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Convert to integers
                conf = float(bbox.conf[0])
                cls = int(bbox.cls[0])
                label = f"{self.yolo_model.names[cls]} {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert processed image to ROS CompressedImage
        _, compressed_img = cv2.imencode('.jpg', image)
        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header  # Keep original timestamp
        compressed_msg.format = "jpeg"
        compressed_msg.data = np.array(compressed_img).tobytes()
        rospy.loginfo(f"Processed Image Shape: {image.shape}")


        # Publish processed image
        self.image_pub.publish(compressed_msg)





if __name__ == '__main__':
    # Create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.loginfo("🚀 Camera reader node with YOLO started, publishing processed images on /yolo_processed_image")

    # Keep spinning
    rospy.spin()

