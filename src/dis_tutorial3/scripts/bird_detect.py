#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
from torchvision import transforms, models
import cv2
import numpy as np
import os

from dis_tutorial3.msg import DetectedBird  # Custom message

class BirdDetector(Node):
    def __init__(self):
        super().__init__('detect_birds')

        # Paths
        self.yolo_model = YOLO("model/bird_yolov8n.pt")  # replace with your detector
        self.resnet_model_path = "model/bird_species_resnet18.pt"
        self.data_dir = "train_bird_classifier/filtered_data"  # for class names

        self.bridge = CvBridge()

        # ROS I/O
        self.sub_image = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, 10)
        self.bird_pub = self.create_publisher(DetectedBird, "/detected_birds", 10)

        # Setup classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 24)
        self.resnet.load_state_dict(torch.load(self.resnet_model_path, map_location=self.device))
        self.resnet.to(self.device)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # Get class index → name mapping
        self.class_names = sorted(os.listdir(self.data_dir))

        self.get_logger().info("🦜 Bird detection + classification node ready")

    def image_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.yolo_model.predict(source=[img_bgr], conf=0.7, save=False)

            boxes = results[0].boxes
            last_crop = None  # 👈 track last crop

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                last_crop = crop  # 👈 update crop for preview

                # Convert crop to PIL then tensor
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = self.transform(transforms.ToPILImage()(crop_rgb)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred = self.resnet(crop_pil)
                    class_idx = int(pred.argmax(1).item())
                    class_name = self.class_names[class_idx]
                    class_conf = torch.softmax(pred, dim=1)[0][class_idx].item()
                if class_conf < 0.4:
                    self.get_logger().info(f"🔸 Skipped {class_name} (low confidence: {class_conf:.2f})")
                    continue


                # Publish result
                out_msg = DetectedBird()
                out_msg.header = Header()
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = msg.header.frame_id

                out_msg.bbox_xmin = x1
                out_msg.bbox_ymin = y1
                out_msg.bbox_xmax = x2
                out_msg.bbox_ymax = y2
                out_msg.center_x = (x1 + x2) // 2
                out_msg.center_y = (y1 + y2) // 2
                out_msg.class_name = class_name
                out_msg.confidence = float(class_conf)

                self.bird_pub.publish(out_msg)
                self.get_logger().info(f"🟢 Detected {class_name} ({class_conf:.2f}) at [{x1},{y1},{x2},{y2}]")

            # === Live preview ===
            if last_crop is not None:
                resized_crop = cv2.resize(last_crop, (224, 224))
                cv2.imshow("Last Bird Detected", resized_crop)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"❌ Failed to detect/classify bird: {e}")


    
def main(args=None):
    rclpy.init(args=args)
    node = BirdDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

