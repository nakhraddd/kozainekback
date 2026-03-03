from typing import Protocol, List
import numpy as np
from ultralytics import YOLO
from app.domain.models import DetectionResult

class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        pass

class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > self.conf_threshold:
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    coords = (float(box.xyxy[0][0]), float(box.xyxy[0][1]), float(box.xyxy[0][2]), float(box.xyxy[0][3]))
                    detections.append(DetectionResult(name=name, confidence=conf, box_coordinates=coords))
                    
        return detections