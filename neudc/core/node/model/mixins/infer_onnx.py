from neudc.core.communication.messaging.types import Batch
from neudc.nn.backends.base import BackendType, BaseBackend
from neudc.utils import LOGGER, PROFILE_FREQ, Profile
from neudc.utils.types import FloatFeaturesBatch


class InferONNXMixin(InferenceMixin, CollectBatchMixin):
    """Mixin class for ONNX inference."""

    def __init__(self, backend: BaseBackend, *args, **kwargs) -> None:
        """Initialize the ONNX inference mixin.

        Args:
        ----
            backend (BaseBackend): The ONNX backend to use for inference.

        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.backend_type = BackendType.ONNXRUNTIME

    @Profile(use_cuda=self.backend.cuda, use_torch=False, logger=LOGGER, freq=PROFILE_FREQ, name="onnx")
    def _infer(self, batch: Batch) -> FloatFeaturesBatch:
        """Perform inference on the collected batch."""
        return self.backend(batch.frames)
