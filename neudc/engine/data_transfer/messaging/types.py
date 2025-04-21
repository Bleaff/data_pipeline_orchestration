from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel

class BaseAttribute(BaseModel):
    r"""
    Base class for box attributes.
    """
    source_node_id: int = -1

class ClassifiedObject(BaseAttribute):
    r"""
    Attribute showing belonging to a class.
    """
    class_id: str
    score: float = None
class Segmentation(BaseAttribute):
    r"""
    Class image segmentation attribute for box.
    """
    image: np.ndarray
    class Config:
        arbitrary_types_allowed = True

class Text(BaseAttribute):
    r"""
    Text attribute for box.
    """
    text: str
    score: float

class Box(BaseAttribute):
    r"""
    Box with absolute coordinates.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: str
    score: float

    reid: str = -1
    # feature_vector: Optional[torch.Tensor] = torch.zeros(1, dtype=torch.float32)

    class Config:
        arbitrary_types_allowed = True

class Keypoint(BaseModel):
    r"""
    Key point with relative coordinates [0.0 ... 1.0].
    """
    x: float
    y: float
    class_id: str
    score: float

class Keypoints(BaseModel):
    r"""
    Dict [str, `Keypoint`]. Key: str - describes keypoint. Val: `Keypoint`. Used by `DetectionNode` to get model results from model process.
    """
    keypoints: Dict[str, Keypoint] = {}
    description: str = ""



class Frame(BaseModel):
    r"""
    Frame with boxes.
    """
    frame_id: int
    boxes: List[Box]
