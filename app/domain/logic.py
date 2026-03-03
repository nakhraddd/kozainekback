from app.domain.models import ProcessedObject, DetectionResult

class SpatialAnalyzer:
    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height
        self.frame_area = frame_width * frame_height

    def analyze(self, detection: DetectionResult) -> ProcessedObject:
        x1, y1, x2, y2 = detection.box_coordinates
        cx = x1 + ((x2 - x1) / 2)
        
        pos = "по центру"
        if cx < self.w / 3:
            pos = "слева"
        elif cx > 2 * self.w / 3:
            pos = "справа"

        box_width = x2 - x1
        box_height = y2 - y1
        area = box_width * box_height
        ratio = area / self.frame_area

        if ratio > 0.4:
            dist = "близко"
        elif ratio > 0.1:
            dist = "средне"
        else:
            dist = "далеко"

        return ProcessedObject(name=detection.name, position=pos, distance=dist)