from pydantic import BaseModel
from typing import List, Optional


class DetectionResult(BaseModel):
    name: str
    confidence: float
    box_coordinates: tuple[float, float, float, float]
    distance: Optional[float] = None
    mask_points: Optional[List[List[float]]] = None
    track_id: Optional[int] = None  # Added track_id


class ProcessedObject(BaseModel):
    name: str
    position: str
    distance: str
    distance_cm: Optional[float] = None
    normalized_box: tuple[float, float, float, float]
    priority: Optional[int] = None
    normalized_mask_points: Optional[List[List[float]]] = None
    track_id: Optional[int] = None  # Added track_id