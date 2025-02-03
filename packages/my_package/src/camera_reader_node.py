#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO  # Ensure YOLO library is installed

class CameraReaderNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("camera_reader_node", anonymous=True)

        # Get the vehicle name (Duckiebot name) from the environment
        self._vehicle_name = os.environ.get("VEHICLE_NAME", "default_vehicle")
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # Load YOLO model
        model_path = os.path.join(os.getcwd(), "best.pt")  # Ensure correct model path
        if not os.path.exists(model_path):
            rospy.logerr(f"❌ YOLO model not found at {model_path}. Exiting...")
            exit(1)

        self.yolo_model = YOLO(model_path)

        # Create ROS subscribers and publishers
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.image_pub = rospy.Publisher("/yolo_processed_image", CompressedImage, queue_size=1)
        self.raw_image_pub = rospy.Publisher("/yolo_processed_image_raw", Image, queue_size=1)  # For rqt_image_view

        rospy.loginfo("✅ CameraReaderNode initialized successfully!")
        rospy.loginfo(f"🚀 Subscribing to: {self._camera_topic}")
        rospy.loginfo("📤 Publishing to: /yolo_processed_image (Compressed) & /yolo_processed_image_raw (Raw)")

    def callback(self, msg):
        # Convert compressed image to OpenCV format
        cv_image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if cv_image is None or cv_image.size == 0:
            rospy.logwarn("⚠️ Received an empty image! Skipping frame...")
            return

        # Run YOLO inference
        results = self.yolo_model.predict(source=cv_image, save=False, save_txt=False, conf=0.25, imgsz=320)

        # Draw bounding boxes on the image
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Convert to integers
                conf = float(bbox.conf[0])
                cls = int(bbox.cls[0])
                label = f"{self.yolo_model.names[cls]} {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert and publish the **compressed** image
        _, compressed_img = cv2.imencode(".jpg", cv_image)
        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header  # Keep original timestamp
        compressed_msg.format = "jpeg"
        compressed_msg.data = np.array(compressed_img).tobytes()
        self.image_pub.publish(compressed_msg)

        # Convert and publish the **raw** image for rqt_image_view
        raw_img_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.raw_image_pub.publish(raw_img_msg)

        rospy.loginfo("📸 YOLO processed image published!")

if __name__ == "__main__":
    node = CameraReaderNode()
    rospy.spin()

