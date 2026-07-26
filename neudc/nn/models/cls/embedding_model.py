"""Module defining EmbeddingFilter for converting images to flattened embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from torchvision import transforms

from neudc.utils import PROFILE_FREQ, NoProfile, Profile

from .base import BaseClsModel

if TYPE_CHECKING:
    from neudc.nn.backends import BaseBackend
    from neudc.utils.types import (
        UInt8HWC,
    )


class EmbeddingFilter(BaseClsModel):
    """Transform images into embedding vectors and flatten outputs.

    This model loads a truncated ResNet, applies standard ImageNet
    preprocessing, runs inference via a specified backend, and
    returns 1D embeddings.
    """

    def __init__(  # noqa: PLR0917 - classifier config constructor, one flag per tunable
        self,
        path: str,
        backend: type[BaseBackend],
        device_id: int = 0,
        eps: float = 0.015,
        min_samples: int = 10,
        num_extremes: int = 2,
    ) -> None:
        """Initialize the EmbeddingFilter.

        Args:
        ----
            path (str): Path to the TorchScript model file.
            backend (Type[BaseBackend]): Backend class for inference.
            device_id (int, optional): CUDA device index. Defaults to 0.
            eps (float, optional): DBSCAN eps parameter (unused here). Defaults to 0.015.
            min_samples (int, optional): DBSCAN min_samples (unused here). Defaults to 10.
            num_extremes (int, optional): Num extremes per cluster (unused here). Defaults to 2.

        """
        super().__init__(path=path, backend=backend, device_id=device_id)  # type: ignore[safe-super]

        self.backend = backend(path=path, device_id=device_id)

        self.eps = eps
        self.min_samples = min_samples
        self.num_extremes = num_extremes
        self.path = path
        self.device_id = device_id

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
    def get_transform(self) -> transforms.Compose:
        """Build and return a torchvision transform pipeline.

        - Converts CV2 BGR→RGB
        - Resizes shorter side to 256
        - Center-crops to 224x224
        - Converts to Tensor and normalizes to ImageNet means/stds.

        Returns
        -------
            torchvision.transforms.Compose: The composed transform.

        """
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
    def pre_transform(
        self,
        ims: list[UInt8HWC],
    ) -> np.ndarray:
        """Apply get_transform to a list of raw images and stack into a batch.

        Args:
        ----
            ims (List[UInt8HWC]): List of images as HxWxC uint8 NumPy arrays (BGR order).

        Returns:
        -------
            np.ndarray: Batched array of shape (N, 3, 224, 224), dtype=float32.

        """
        transform = self.get_transform()
        transformed_imgs = []
        for im in ims:
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            transformed_img = transform(img).numpy()
            transformed_imgs.append(transformed_img)

        return np.stack(transformed_imgs, axis=0)

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
    def post_transform(
        self,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Flatten the network outputs into 2D embedding vectors.

        Args:
        ----
            predictions (np.ndarray): Raw model output of shape
                                      (N, C, 1, 1) or similar.

        Returns:
        -------
            np.ndarray: Embeddings of shape (N, C) after flattening.

        """
        return predictions.reshape(predictions.shape[0], -1)

    def __call__(self, ims: list[UInt8HWC]) -> np.ndarray:  # type: ignore[override]
        """Run the full embedding filter pipeline on a batch of images.

        Args:
        ----
            ims (List[UInt8HWC]): List of raw images.
            return_embeddings (bool): Ignored; always returns embeddings.

        Returns:
        -------
            np.ndarray: Embedding vectors of shape (N, D).

        """
        batch = self.pre_transform(ims)
        output = self.backend(batch)[0]

        return self.post_transform(output)

    def __repr__(self) -> str:
        """Return the string representation including key parameters."""
        return (
            f"EmbeddingFilter(path={self.path}, device_id={self.device_id}, eps={self.eps}, "
            f"min_samples={self.min_samples}, num_extremes={self.num_extremes})"
        )

    @NoProfile  # type: ignore[call-arg]  # NoProfile is a singleton instance mistyped as a class by mypy
    def warmup(self, iters: int = 10) -> None:
        """No-op; embedding extraction currently has no warmup pass implemented.

        Args:
        ----
            iters (int): Unused; kept for interface compatibility with `BaseClsModel.warmup`.

        """
