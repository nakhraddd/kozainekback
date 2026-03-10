import cv2
from typing import Protocol, List
import numpy as np
from ultralytics import YOLO
from app.domain.models import DetectionResult

class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        pass

# Define known real-world widths for common objects in centimeters.
KNOWN_OBJECT_WIDTHS = {
    "person": 50.0, "bicycle": 170.0, "car": 180.0, "motorcycle": 80.0,
    "airplane": 5000.0, "bus": 250.0, "train": 280.0, "truck": 250.0,
    "boat": 200.0, "traffic light": 30.0, "fire hydrant": 25.0, "stop sign": 75.0,
    "parking meter": 15.0, "bench": 150.0, "bird": 15.0, "cat": 15.0,
    "dog": 25.0, "horse": 60.0, "sheep": 40.0, "cow": 80.0, "elephant": 150.0,
    "bear": 70.0, "zebra": 60.0, "giraffe": 100.0, "backpack": 30.0,
    "umbrella": 90.0, "handbag": 30.0, "tie": 7.0, "suitcase": 50.0,
    "frisbee": 25.0, "skis": 10.0, "snowboard": 25.0, "sports ball": 22.0,
    "kite": 100.0, "baseball bat": 7.0, "baseball glove": 20.0,
    "skateboard": 20.0, "surfboard": 50.0, "tennis racket": 25.0,
    "bottle": 7.0, "wine glass": 8.0, "cup": 8.0, "fork": 2.5, "knife": 2.0,
    "spoon": 3.0, "bowl": 15.0, "banana": 3.0, "apple": 8.0, "sandwich": 10.0,
    "orange": 8.0, "broccoli": 15.0, "carrot": 3.0, "hot dog": 4.0,
    "pizza": 30.0, "donut": 9.0, "cake": 20.0, "chair": 45.0, "couch": 200.0,
    "potted plant": 30.0, "bed": 150.0, "dining table": 100.0, "toilet": 40.0,
    "tv": 100.0, "laptop": 35.0, "mouse": 7.0, "remote": 5.0, "keyboard": 40.0,
    "cell phone": 7.0, "microwave": 50.0, "oven": 60.0, "toaster": 25.0,
    "sink": 50.0, "refrigerator": 70.0, "book": 15.0, "clock": 30.0,
    "vase": 15.0, "scissors": 8.0, "teddy bear": 30.0, "hair drier": 10.0,
    "toothbrush": 2.0, "A4 paper": 21.0, "stairs": 100.0
}

# English to Russian translation dictionary
RUSSIAN_NAMES = {
    "person": "человек", "bicycle": "велосипед", "car": "машина", "motorcycle": "мотоцикл",
    "airplane": "самолет", "bus": "автобус", "train": "поезд", "truck": "грузовик",
    "boat": "лодка", "traffic light": "светофор", "fire hydrant": "пожарный гидрант",
    "stop sign": "знак стоп", "parking meter": "паркомат", "bench": "скамейка",
    "bird": "птица", "cat": "кошка", "dog": "собака", "horse": "лошадь", "sheep": "овца",
    "cow": "корова", "elephant": "слон", "bear": "медведь", "zebra": "зебра",
    "giraffe": "жираф", "backpack": "рюкзак", "umbrella": "зонт", "handbag": "сумка",
    "tie": "галстук", "suitcase": "чемодан", "frisbee": "фризби", "skis": "лыжи",
    "snowboard": "сноуборд", "sports ball": "спортивный мяч", "kite": "воздушный змей",
    "baseball bat": "бейсбольная бита", "baseball glove": "бейсбольная перчатка",
    "skateboard": "скейтборд", "surfboard": "доска для серфинга", "tennis racket": "теннисная ракетка",
    "bottle": "бутылка", "wine glass": "бокал", "cup": "чашка", "fork": "вилка",
    "knife": "нож", "spoon": "ложка", "bowl": "миска", "banana": "банан", "apple": "яблоко",
    "sandwich": "бутерброд", "orange": "апельсин", "broccoli": "брокколи", "carrot": "морковь",
    "hot dog": "хот-дог", "pizza": "пицца", "donut": "пончик", "cake": "торт",
    "chair": "стул", "couch": "диван", "potted plant": "растение в горшке", "bed": "кровать",
    "dining table": "обеденный стол", "toilet": "туалет", "tv": "телевизор", "laptop": "ноутбук",
    "mouse": "мышь", "remote": "пульт", "keyboard": "клавиатура", "cell phone": "телефон",
    "microwave": "микроволновка", "oven": "духовка", "toaster": "тостер", "sink": "раковина",
    "refrigerator": "холодильник", "book": "книга", "clock": "часы", "vase": "ваза",
    "scissors": "ножницы", "teddy bear": "плюшевый мишка", "hair drier": "фен",
    "toothbrush": "зубная щетка", "A4 paper": "бумага А4", "stairs": "лестница"
}

class YoloDetector:
    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        conf_threshold: float = 0.5,
        focal_length: float = 1680,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.focal_length = focal_length

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        # --- Canny + Hough Transform for obstacle detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        h, w = frame.shape[:2]
        roi_w_start = w // 3
        roi_w_end = 2 * w // 3
        roi = edges[:, roi_w_start:roi_w_end]
        
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=100, minLineLength=h*0.3, maxLineGap=20)
        
        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 10:
                    vertical_lines.append(line)

        obstacle_detections = []
        if len(vertical_lines) > 0:
            obstacle_box = (float(roi_w_start), 0.0, float(roi_w_end), float(h))
            obstacle_detections.append(
                DetectionResult(
                    name="Caution! Possible obstacle ahead",
                    confidence=0.9,
                    box_coordinates=obstacle_box,
                    distance=None,
                    mask_points=[],
                    track_id=None
                )
            )

        # --- Stairs detection ---
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=w*0.4, maxLineGap=20)
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10:
                    horizontal_lines.append(line)
        
        if len(horizontal_lines) > 4:
            stairs_box = (0.0, 0.0, float(w), float(h))
            obstacle_detections.append(
                DetectionResult(
                    name="stairs",
                    confidence=0.95,
                    box_coordinates=stairs_box,
                    distance=None,
                    mask_points=[],
                    track_id=None
                )
            )

        # --- YOLO object detection ---
        results = self.model.track(
            frame, 
            persist=True,
            verbose=False,
            rect=True,
            imgsz=640,
            conf=self.conf_threshold,
            iou=0.7
        )
        yolo_detections = []
        
        for r in results:
            img_height, img_width = r.orig_shape
            
            if r.boxes.id is None:
                continue

            for i, box in enumerate(r.boxes):
                track_id = int(box.id[0])
                conf = float(box.conf[0])
                
                cls_id = int(box.cls[0])
                english_name = self.model.names[cls_id]
                
                name = RUSSIAN_NAMES.get(english_name, english_name)
                
                x1, y1, x2, y2 = box.xyxy[0]
                coords = (float(x1), float(y1), float(x2), float(y2))
                
                distance = None
                known_width = KNOWN_OBJECT_WIDTHS.get(english_name)
                if known_width is not None:
                    pixel_width = x2 - x1
                    if pixel_width > 0:
                        distance = (known_width * self.focal_length) / pixel_width

                mask_points = []
                if r.masks and i < len(r.masks):
                    mask = r.masks[i]
                    if len(mask.xy) > 0:
                        polygon = mask.xy[0]
                        if polygon.size > 0:
                            normalized_polygon = polygon / np.array([img_width, img_height])
                            mask_points = normalized_polygon.tolist()
                
                yolo_detections.append(
                    DetectionResult(
                        name=name,
                        confidence=conf,
                        box_coordinates=coords,
                        distance=distance,
                        mask_points=mask_points,
                        track_id=track_id
                    )
                )
                    
        return obstacle_detections + yolo_detections