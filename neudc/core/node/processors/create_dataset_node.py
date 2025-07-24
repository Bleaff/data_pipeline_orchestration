from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.processors.mixins.create_dataset_mixin import CreateDatasetMixin

if TYPE_CHECKING:
    pass


class CreateDataset(CreateDatasetMixin, BaseThreadedNode):
    """
    A threaded processing node that saves incoming image frames and corresponding YOLO-format
    annotations to disk for dataset creation.

    This node is part of a data collection or labeling pipeline. It combines:
    - `CreateDatasetMixin` for saving frames and annotations.
    - `BaseThreadedNode` to run as a threaded processing unit in the pipeline.

    Attributes:
        save_dir (str): Root directory for saving images and label files.
        mailbox (Any): Communication interface for message passing.
    """

    def __init__(self, mailbox: Any, save_dir: str) -> CreateDataset:
        """
        Initializes the CreateDataset node.

        Args:
            mailbox (Any): A mailbox object used for message communication in the pipeline.
            save_dir (str): Path to the directory where images and labels will be saved.
        """
        super().__init__(save_dir=save_dir, mailbox=mailbox)

    @classmethod
    def from_config(cls: CreateDataset, config: dict[str, Any]) -> CreateDataset:
        """
        Creates a CreateDataset instance from a configuration dictionary.

        Args:
            config (dict[str, Any]): A dictionary containing keys:
                - 'mailbox': Mailbox instance for message passing.
                - 'save_dir': Path to the directory where images and labels will be saved.

        Returns:
            CreateDataset: An initialized instance of the node.
        """

        return CreateDataset(
            mailbox=config["mailbox"],
            save_dir=config["save_dir"],
        )
