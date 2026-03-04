from pydantic import BaseModel

class DetectionResult(BaseModel):
    name: str
    confidence: float
    box_coordinates: tuple[float, float, float, float]
    distance: float | None = None
    
class ProcessedObject(BaseModel):
    name: str
    position: str
    distance: str
    distance_cm: float | None = None
    normalized_box: tuple[float, float, float, float]