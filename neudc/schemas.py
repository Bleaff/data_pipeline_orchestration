import os
from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field, field_validator

from neudc.nn.backends import BackendType
from neudc.utils import LOGGER

__all__ = ("Config",)


class ModelConfig(BaseModel):
    path: str
    device: str = "cpu"
    backend: BackendType = BackendType.TORCH  # Default to TorchBackend
    conf: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    iou: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    imgsz: Annotated[Union[int, tuple[int, int]], Field(gt=0)] = 640
    names: Annotated[list[str], Field(min_length=1)]

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda"):
            raise ValueError("Device must be either 'cpu' or 'cuda'")
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: BackendType, values) -> BackendType:
        # If backend is TensorRT, device must be cuda
        if v == BackendType.TENSORRT:
            device = values.data.get("device")
            if "cuda" not in device:
                raise ValueError("TensorRTBackend only supports 'cuda' device")
        return v

    @field_validator("path")
    @classmethod
    def validate_model_path(cls, v: str, values) -> str:
        # Ensure the file exists
        if not os.path.exists(v):
            raise ValueError(f"Model path {v} does not exist")

        # Check file extension
        if not v.lower().endswith((".pt", ".pth", ".engine", ".torchscript")):
            raise ValueError("Model file must be a .pt, .pth, .torchscript or .engine file")

        # Additional backend-specific validation
        backend = values.data.get("backend")
        if backend == BackendType.TORCH and not v.lower().endswith((".pt", ".pth", ".torchscript")):
            raise ValueError("For TorchBackend, model file must be a .pt, .torchscript or .pth file")
        elif backend == BackendType.TENSORRT and not v.lower().endswith(".engine"):
            raise ValueError("For TensorRTBackend, model file must be a .engine file")

        return v

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, v: Union[int, tuple[int, int]]) -> Union[int, tuple[int, int]]:
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("If imgsz is a list, it must contain exactly two elements [h, w]")
            if not all(isinstance(x, int) and x > 0 for x in v):
                raise ValueError("Both elements in imgsz list must be positive integers")
        elif isinstance(v, int):
            if v <= 0:
                raise ValueError("imgsz must be a positive integer")
        else:
            raise ValueError("imgsz must be either an integer or a list of two integers")
        return v


class DataConfig(BaseModel):
    paths: list[str]
    skip_frames: Annotated[int, Field(ge=0)] = 0
    intervals: Optional[
        Annotated[list[Annotated[list[float], Field(min_items=2, max_items=2)]], Field(min_length=1)]
    ] = None
    save_result_dir: str = "results/"
    save_imgs_with_prediction_bbox: bool = True
    save_coco_annotations: bool = True
    save_zip: bool = False
    check_duplicates: bool = False

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, v: list[str]) -> list[str]:
        paths = []
        for path in v:
            if not os.path.exists(v):
                LOGGER.info(f"Inference path {path} doesn't exist, skip.")
            else:
                paths.append(path)
        return paths

    @field_validator("intervals")
    @classmethod
    def validate_intervals(cls, v: list[list[float]]) -> list[list[float]]:
        for interval in v:
            if len(interval) != 2:
                raise ValueError("Interval must contain exactly 2 values")
            if interval[0] >= interval[1]:
                raise ValueError("Start time must be before end time")
        return v

    @field_validator("save_result_dir")
    @classmethod
    def validate_result_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
