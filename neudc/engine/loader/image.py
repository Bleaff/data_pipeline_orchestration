import os

import cv2

from neudc.utils import LOGGER

from .base import BaseLoader

__all__ = ("ImageLoader",)


class ImageLoader(BaseLoader):

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    """
    Iterates over all images in a specified directory and
    returns (file_name, frame) when iterated.
    """

    def __init__(self, path: str) -> "ImageLoader":
        self.images_dir = path
        self._index = 0  # Tracks the current position in the iteration
        self.image_files = self._find_files(path)

        LOGGER.info(f"Added ImageLoader: path={path}, files={len(self.image_files)}")

    def _find_files(self, directory: str) -> list[str]:
        """
        Recursively finds all image files in the directory with valid extensions.
        """
        image_files = []

        if not os.path.isdir(directory):
            LOGGER.error(f"{directory} is not valid for ImageLoader!")
            return image_files

        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.lower().endswith(ImageLoader.valid_extensions):
                    image_files.append(file_path)
                else:
                    LOGGER.warning(f"WARNING ⚠️ Extensions for the file {file_path} is not supported.")

        return image_files

    def __iter__(self):
        for file_path in self.image_files:
            if not os.path.exists(file_path):
                LOGGER.warning(f"WARNING ⚠️ Path {file_path} doesn't exist.")
                continue

            # Read the image
            img = cv2.imread(file_path)
            if img is None:
                LOGGER.warning(f"WARNING ⚠️ Can't read the image {file_path}.")
                continue

            yield file_path, img

    def __len__(self) -> int:
        return len(self.image_files)
