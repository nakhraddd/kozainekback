from pydantic import BaseModel

class DetectionResult(BaseModel):
    name: str
    confidence: float
    box_coordinates: tuple[float, float, float, float]
    
class ProcessedObject(BaseModel):
    name: str
    position: str
    distance: str
    normalized_box: tuple[float, float, float, float]