from typing import Protocol, List
import numpy as np
from ultralytics import YOLO
from app.domain.models import DetectionResult

class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        pass

# Define known real-world widths for common objects in centimeters.
# You should expand and refine this dictionary with the objects you need to detect
# and their accurate average dimensions.
KNOWN_OBJECT_WIDTHS = {
    "person": 50.0,  # Average shoulder width of an adult
    "bicycle": 170.0, # Average length of a bicycle
    "car": 180.0,    # Average width of a car
    "motorcycle": 80.0, # Average width of a motorcycle
    "airplane": 5000.0, # Very large, depends on type
    "bus": 250.0,    # Average width of a bus
    "train": 280.0,  # Average width of a train car
    "truck": 250.0,  # Average width of a truck
    "boat": 200.0,   # Highly variable
    "traffic light": 30.0, # Average width
    "fire hydrant": 25.0, # Average width
    "stop sign": 75.0,   # Average width
    "parking meter": 15.0, # Average width
    "bench": 150.0,  # Average length
    "bird": 15.0,    # Highly variable, average wingspan
    "cat": 15.0,     # Average width
    "dog": 25.0,     # Average width
    "horse": 60.0,   # Average width
    "sheep": 40.0,   # Average width
    "cow": 80.0,     # Average width
    "elephant": 150.0, # Average width
    "bear": 70.0,    # Average width
    "zebra": 60.0,   # Average width
    "giraffe": 100.0, # Average width
    "backpack": 30.0, # Average width
    "umbrella": 90.0, # Average diameter when open
    "handbag": 30.0, # Average width
    "tie": 7.0,      # Average width
    "suitcase": 50.0, # Average width
    "frisbee": 25.0, # Average diameter
    "skis": 10.0,    # Average width
    "snowboard": 25.0, # Average width
    "sports ball": 22.0, # Average diameter (e.g., soccer ball)
    "kite": 100.0,   # Highly variable
    "baseball bat": 7.0, # Average diameter
    "baseball glove": 20.0, # Average width
    "skateboard": 20.0, # Average width
    "surfboard": 50.0, # Average width
    "tennis racket": 25.0, # Average width
    "bottle": 7.0,   # Average diameter
    "wine glass": 8.0, # Average diameter
    "cup": 8.0,      # Average diameter
    "fork": 2.5,     # Average width
    "knife": 2.0,    # Average width
    "spoon": 3.0,    # Average width
    "bowl": 15.0,    # Average diameter
    "banana": 3.0,   # Average width
    "apple": 8.0,    # Average diameter
    "sandwich": 10.0, # Average width
    "orange": 8.0,   # Average diameter
    "broccoli": 15.0, # Average width
    "carrot": 3.0,   # Average width
    "hot dog": 4.0,  # Average width
    "pizza": 30.0,   # Average diameter
    "donut": 9.0,    # Average diameter
    "cake": 20.0,    # Average diameter
    "chair": 45.0,   # Average width
    "couch": 200.0,  # Average length
    "potted plant": 30.0, # Average width
    "bed": 150.0,    # Average width
    "dining table": 100.0, # Average width
    "toilet": 40.0,  # Average width
    "tv": 100.0,     # Highly variable, average width
    "laptop": 35.0,  # Average width
    "mouse": 7.0,    # Average width
    "remote": 5.0,   # Average width
    "keyboard": 40.0, # Average width
    "cell phone": 7.0, # Average width
    "microwave": 50.0, # Average width
    "oven": 60.0,    # Average width
    "toaster": 25.0, # Average width
    "sink": 50.0,    # Average width
    "refrigerator": 70.0, # Average width
    "book": 15.0,    # Average width
    "clock": 30.0,   # Average diameter
    "vase": 15.0,    # Average width
    "scissors": 8.0, # Average width
    "teddy bear": 30.0, # Average width
    "hair drier": 10.0, # Average width
    "toothbrush": 2.0, # Average width
    "A4 paper": 21.0 # Specific for A4 paper
}


class YoloDetector:
    # To calibrate the focal length, place an object with a known width (in cm)
    # at a known distance (in cm) from the camera.
    # Then, use the formula: focal_length = (pixel_width * known_distance) / known_width
    def __init__(
        self,
        model_path: str = "yolov8s.pt", # Changed to yolov8s.pt
        conf_threshold: float = 0.5,
        focal_length: float = 840,  # Example focal length, needs calibration
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
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
                    
                    distance = None # Initialize distance as None
                    
                    # Get known_width for the detected object
                    known_width = KNOWN_OBJECT_WIDTHS.get(name)
                    
                    if known_width is not None:
                        # Calculate distance only if known_width is available
                        pixel_width = x2 - x1
                        if pixel_width > 0: # Avoid division by zero
                            distance = (known_width * self.focal_length) / pixel_width
                    
                    detections.append(
                        DetectionResult(
                            name=name,
                            confidence=conf,
                            box_coordinates=coords,
                            distance=float(distance) if distance is not None else None,
                        )
                    )
                    
        return detections