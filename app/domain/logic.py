from app.domain.models import ProcessedObject, DetectionResult

class SpatialAnalyzer:
    def __init__(self, frame_width: int, frame_height: int, reference_distance: float = 100.0):
        self.w = frame_width
        self.h = frame_height
        self.frame_area = frame_width * frame_height
        self.reference_distance = reference_distance

    def analyze(self, detection: DetectionResult) -> ProcessedObject:
        x1, y1, x2, y2 = detection.box_coordinates
        cx = x1 + ((x2 - x1) / 2)
        
        pos = "по центру"
        if cx < self.w / 3:
            pos = "слева"
        elif cx > 2 * self.w / 3:
            pos = "справа"

        distance_cm = detection.distance
        dist_str = "неизвестно"

        if distance_cm is not None:
            if distance_cm < 0.7 * self.reference_distance:
                dist_str = "близко"
            elif distance_cm <= 1.5 * self.reference_distance:
                dist_str = "средне"
            else:
                dist_str = "далеко"

        norm_box = (x1 / self.w, y1 / self.h, x2 / self.w, y2 / self.h)

        normalized_mask_points = detection.mask_points

        return ProcessedObject(
            name=detection.name, 
            position=pos, 
            distance=dist_str, 
            distance_cm=distance_cm,
            normalized_box=norm_box,
            normalized_mask_points=normalized_mask_points,
            track_id=detection.track_id # Added track_id
        )