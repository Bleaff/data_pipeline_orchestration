from typing import Any, Optional

from neudc.core.communication.messaging.types import Frame
from neudc.core.node.model.base_batch_process_inference import BaseBatchProcessInference


class ActiveLearning(BaseBatchProcessInference):
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

    def postprocess_result(self):
        pass