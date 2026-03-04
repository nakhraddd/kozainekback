from typing import Protocol, List
import numpy as np
from ultralytics import YOLO
from app.domain.models import DetectionResult

class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        pass

class YoloDetector:
    # To calibrate the focal length, place an object with a known width (in cm)
    # at a known distance (in cm) from the camera.
    # Then, use the formula: focal_length = (pixel_width * known_distance) / known_width
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        known_width: float = 21.0,  # A4 paper width in cm
        focal_length: float = 840,  # Example focal length
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.known_width = known_width
        self.focal_length = focal_length

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > self.conf_threshold:
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0]
                    coords = (float(x1), float(y1), float(x2), float(y2))
                    
                    # Calculate distance
                    pixel_width = x2 - x1
                    distance = (self.known_width * self.focal_length) / pixel_width
                    
                    detections.append(
                        DetectionResult(
                            name=name,
                            confidence=conf,
                            box_coordinates=coords,
                            distance=float(distance),
                        )
                    )
                    
        return detections