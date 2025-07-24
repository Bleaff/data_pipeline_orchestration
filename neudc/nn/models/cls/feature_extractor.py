"""Feature extraction module for image classification.

Contains functions for resizing, block processing, entropy filtering, etc.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import square


class FeatureExtractor:
    """A feature extraction class for blur detection using DCT (Discrete Cosine Transform) analysis.

    This class implements a blur detection algorithm that:
    1. Computes regions of interest (ROI) using local entropy filtering
    2. Extracts DCT coefficients from image blocks
    3. Analyzes high-frequency components to determine blur characteristics

    The algorithm is based on the principle that blurry images have fewer high-frequency
    components compared to sharp images.

    Attributes
    ----------
        blockSize_feature_extractor (int): Size of image blocks for feature extraction (32x32)
        downsamplingFactor (int): Factor for image downsampling to improve processing speed
        entropy_filt_kernel_sze (int): Kernel size for local entropy filtering
        local_entropy_thresh (float): Threshold for ROI detection based on local entropy
        valid_img_block_thresh (float): Threshold for determining valid image blocks
        resized_image (np.ndarray): Downsampled version of the input image
        roi (np.ndarray): Binary mask indicating regions of interest

    """

    def __init__(self) -> None:
        """Initialize the FeatureExtractor object with default parameters for blur detection.

        This constructor sets up the necessary configuration for feature extraction:
        - Defines block size for DCT-based analysis
        - Sets the downsampling factor for speed
        - Establishes thresholds for entropy-based ROI filtering
        - Precomputes DCT matrices and frequency band masks

        The class is designed to process grayscale images for sharpness/blur analysis,
        using localized DCT coefficient patterns extracted from regions with sufficient texture.

        Attributes initialized:
        ------------------------
        - blockSize_feature_extractor: Size of image blocks used for DCT analysis
        - downsamplingFactor: Factor by which the input image is resized
        - entropy_filt_kernel_sze: Size of the kernel used in entropy filtering
        - local_entropy_thresh: Threshold for deciding if a region has enough texture
        - valid_img_block_thresh: Threshold for validating an image block based on ROI
        - resized_image: Placeholder for resized image
        - roi: Placeholder for region of interest mask
        - __dct_matrices: Precomputed DCT matrix
        - __freqBands: Frequency masks used to select relevant DCT bands
        """
        self.blockSize_feature_extractor = 32
        self.downsamplingFactor = 2
        self.resized_image: list[Any] = []
        self.entropy_filt_kernel_sze = 16
        self.local_entropy_thresh = 0.6
        self.valid_img_block_thresh = 0.7
        self.roi: list[Any] = []
        self.__freqBands: list[Any] = []
        self.__dct_matrices = self.__dctmtx(self.blockSize_feature_extractor)
        self.__compute_frequency_bands()

    def __dctmtx(self, n: int) -> np.ndarray:
        """Generate the DCT (Discrete Cosine Transform) transformation matrix.

        Creates an orthogonal DCT matrix used for transforming image blocks into
        frequency domain for blur analysis.

        Args:
        ----
            n (int): Size of the square DCT matrix (typically 32 for 32x32 blocks)

        Returns:
        -------
            np.ndarray: DCT transformation matrix of size n x n

        Note:
        ----
            - Uses Type-II DCT with normalization
            - First row is scaled by 1/sqrt(2) for orthogonality

        """
        [mesh_cols, mesh_rows] = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))
        dct_matrix = np.sqrt(2 / n) * np.cos(np.pi * np.multiply((2 * mesh_cols + 1), mesh_rows) / (2 * n))
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return dct_matrix

    def __compute_frequency_bands(self) -> None:
        """Compute frequency band masks for DCT coefficient analysis.

        Creates masks to identify different frequency regions in the DCT domain:
        - Band 0: DC and very low frequencies (excluded from analysis)
        - Band 1: Low to medium frequencies
        - Band 2: High frequencies (most important for blur detection)
        - Band 3: DC component (set separately)

        Note:
        ----
            - High-frequency components (band 2) are most affected by blur
            - The algorithm focuses on these components for blur detection

        """
        current_scale = self.blockSize_feature_extractor
        matrix_inds = np.zeros((current_scale, current_scale))

        for i in range(current_scale):
            matrix_inds[0 : max(0, int(((current_scale - 1) / 2) - i + 1)), i] = 1

        for i in range(current_scale):
            if (current_scale - ((current_scale - 1) / 2) - i) <= 0:
                matrix_inds[0 : current_scale - i - 1, i] = 2
            else:
                matrix_inds[int(current_scale - ((current_scale - 1) / 2) - i - 1) : int(current_scale - i - 1), i] = 2
        matrix_inds[0, 0] = 3
        self.__freqBands.append(matrix_inds)

    def resize_image(self, img: np.ndarray, rows: int, cols: int) -> None:
        """Resize the input image for efficient processing.

        Downsamples the image by the specified factor to reduce computational complexity
        while preserving blur detection accuracy.

        Args:
        ----
            img (np.ndarray): Input grayscale image
            rows (int): Original image height
            cols (int): Original image width

        Note:
        ----
            - Downsampling improves processing speed
            - The downsampling factor is typically 2 for good speed/accuracy trade-off
            - Resized image is stored in self.resized_image

        """
        self.resized_image = cv2.resize(img, (int(cols / self.downsamplingFactor), int(rows / self.downsamplingFactor)))
        rows = np.shape(self.resized_image)[0]
        cols = np.shape(self.resized_image)[1]

    def compute_roi(self) -> None:
        """Compute regions of interest (ROI) using local entropy filtering.

        Identifies image regions with sufficient texture and detail for reliable
        blur analysis. Areas with low entropy (uniform regions) are excluded
        as they don't provide meaningful blur information.

        Note:
        ----
            - Uses local entropy to measure texture complexity
            - Only regions above the entropy threshold are considered
            - ROI is stored as a binary mask in self.roi

        """
        local_entropy = self.entropy_filt(self.resized_image)
        self.roi = 1.0 * (local_entropy > self.local_entropy_thresh * np.max(local_entropy))

    def get_single_resolution_features(self, block: np.ndarray) -> list[float]:
        """Extract blur features from a single image block using DCT analysis.

        Computes DCT coefficients for the block and extracts high-frequency components
        that are most indicative of image sharpness/blur.

        Args:
        ----
            block (np.ndarray): Image block of size blockSize_feature_extractor x blockSize_feature_extractor

        Returns:
        -------
            List[float]: Sorted list of high-frequency DCT coefficients

        Note:
        ----
            - High-frequency components are reduced in blurry images
            - Features are sorted to provide consistent ordering
            - Only frequency band 0 components are used (high frequencies)

        """
        d = self.__dct_matrices
        dct_coeff = np.abs(np.matmul(np.matmul(d, block), np.transpose(d)))
        temp = np.where(self.__freqBands[0] == 0)
        high_freq_components = dct_coeff[temp]
        return sorted(high_freq_components)

    def extract_feature(self) -> list[list[float]]:
        """Extract blur features from the entire image.

        Processes the image in blocks, extracting DCT-based features from each
        valid block within the computed regions of interest.

        Returns:
        -------
            List[List[float]]: List of feature vectors, where each vector contains
                              DCT coefficients from one image block

        Note:
        ----
            - Only processes blocks that fall within ROI and meet validity criteria
            - Each feature vector represents high-frequency content of one image block
            - Empty list indicates no valid features could be extracted

        """
        extracted_features = []
        rows = np.shape(self.resized_image)[0]
        cols = np.shape(self.resized_image)[1]
        for i in range(0, rows, self.blockSize_feature_extractor):
            for j in range(0, cols, self.blockSize_feature_extractor):
                if self.is_image_block_valid(i, j):
                    block = np.array(self.resized_image)[
                        i : i + self.blockSize_feature_extractor,
                        j : j + self.blockSize_feature_extractor,
                    ]
                    if (np.shape(block)[0] == self.blockSize_feature_extractor) and (
                        np.shape(block)[1] == self.blockSize_feature_extractor
                    ):
                        features = self.get_single_resolution_features(block)
                        extracted_features.append(features)
        return extracted_features

    def is_image_block_valid(self, i: int, j: int) -> bool:
        """Determine if an image block contains sufficient texture for analysis.

        Checks if the block has enough non-uniform content (based on ROI) to
        provide reliable blur detection features.

        Args:
        ----
            i (int): Row index of the block's top-left corner
            j (int): Column index of the block's top-left corner

        Returns:
        -------
            bool: True if the block is valid for feature extraction, False otherwise

        Note:
        ----
            - Uses the ROI mask to determine block validity
            - Blocks with insufficient texture are skipped
            - Threshold is set by valid_img_block_thresh (typically 0.7)

        """
        block = np.array(self.roi)[i : i + self.blockSize_feature_extractor, j : j + self.blockSize_feature_extractor]
        val = np.sum(block) / np.prod(np.shape(block))
        return val > self.valid_img_block_thresh

    def entropy_filt(self, img: np.ndarray) -> np.ndarray:
        """Apply local entropy filtering to measure texture complexity.

        Computes local entropy for each pixel using a square kernel to identify
        regions with sufficient texture for blur analysis.

        Args:
        ----
            img (np.ndarray): Input grayscale image

        Returns:
        -------
            np.ndarray: Local entropy map of the same size as input image

        Note:
        ----
            - Higher entropy indicates more texture/detail
            - Used for ROI computation and block validation
            - Kernel size affects the locality of entropy measurement

        """
        return entropy(img, square(self.entropy_filt_kernel_sze))

    def clear_object(self) -> None:
        """Clear internal state and free memory.

        Resets all stored arrays and computed data to prepare for processing
        a new image or to free memory when the object is no longer needed.

        Note:
        ----
            - Call this method between processing different images
            - Helps prevent memory leaks in long-running applications
            - Does not reset configuration parameters

        """
        self.resized_image = []
        self.roi = []
        self.__freqBands = []
