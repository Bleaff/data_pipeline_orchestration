from typing import Any, Optional

from neudc.core.communication.messaging.types import Frame
from neudc.core.node.model.base_process_inference import BaseProcessInference


class ProcessBlurInference(BaseProcessInference):
    """A processing node for blur inference that handles post-processing of blur detection results.

    This class extends BaseProcessInference and is responsible for interpreting blur detection
    results and deciding whether to pass frames through the pipeline based on blur detection.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ProcessBlurInference node.

        Args:
        ----
            *args: Variable length argument list passed to parent class
            **kwargs: Arbitrary keyword arguments passed to parent class

        """
        super().__init__(*args, **kwargs)

    def postprocess_result(self, result: Any, item: Frame) -> Optional[Frame]:
        """Post-process blur detection results and determine frame filtering.

        This method examines the blur detection results and decides whether to pass
        the frame through the pipeline or filter it out based on blur detection.

        Args:
        ----
            result (Any): Blur detection results from the model. Expected to be a nested list
                         where result[0] contains boolean values indicating blur detection.
            item (Frame): The input frame/item to be processed

        Returns:
        -------
            Optional[Frame]: Returns the input frame if blur is detected, None if no blur
                           is detected (filtering out sharp images)

        Note:
        ----
            - If any region in the first image is detected as blurry, the frame is passed through
            - Sharp images are filtered out by returning None
            - This behavior assumes the pipeline is designed to process only blurry images

        """
        for res in result:
            if res[0]:
                return None
            return item
        return None
