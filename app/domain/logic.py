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

        # Use the distance calculated by the detector
        distance_cm = detection.distance
        
        # Determine descriptive distance string based on distance_cm
        if distance_cm is not None:
            if distance_cm < 50:
                dist_str = "близко"
            elif distance_cm < 200:
                dist_str = "средне"
            else:
                dist_str = "далеко"
        else:
            dist_str = "неизвестно" # Fallback if distance is not calculated

        norm_box = (x1 / self.w, y1 / self.h, x2 / self.w, y2 / self.h)

        return ProcessedObject(
            name=detection.name, 
            position=pos, 
            distance=dist_str, # Keep the descriptive string
            distance_cm=distance_cm, # Add the numerical distance
            normalized_box=norm_box
        )