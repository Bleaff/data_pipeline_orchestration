from neudc.core.communication.messaging.types import Frame
from collections import defaultdict
from typing import Optional, Dict, Union, Callable
import imagehash
import hashlib
import numpy as np
from PIL import Image

class HashFilterMixin:
    """
    Mixin for filtering out near-duplicate frames based on different hash algorithms.

    Attributes:
        cache: Mapping from source_frame (str) to a dict of frame_id (int) → hash
               (either imagehash.ImageHash or MD5 hex string).
        delta: Maximum Hamming distance for perceptual hashes to be considered duplicates.
        type_of_hash: One of 'ahash', 'dhash', 'phash', or 'md5'.
        hash_funcs: Mapping from hash_type to the corresponding hashing method.
    """

    def __init__(self, delta: int = 5, hash_type: str = 'ahash', *args, **kwargs) -> None:
        """
        Initialize the mixin.

        Args:
            delta: Maximum Hamming distance for perceptual hash comparisons.
            hash_type: Which hash to compute—'ahash', 'dhash', 'phash', or 'md5'.
        """
        super().__init__(*args, **kwargs)
        self.cache: Dict[str, Dict[int, Union[imagehash.ImageHash, str]]] = defaultdict(dict)
        self.delta: int = delta
        self.type_of_hash: str = hash_type
        self.hash_funcs = {
            'ahash': self.ahash_lib,
            'dhash': self.dhash_lib,
            'phash': self.phash_lib,
            'md5': self.md5_lib,
        }
    def to_pil(self, img: np.ndarray) -> Image.Image:
        """
        Convert to PIL from np
        """
        return Image.fromarray(img)
    
    def ahash_lib(self, frame:Frame, hash_size: int = 8) -> imagehash.ImageHash:
        """
        Compute the average hash (aHash) of the frame.

        Args:
            frame: Frame object containing .image (np.ndarray).
            hash_size: Size parameter for the hash algorithm.

        Returns:
            An imagehash.ImageHash instance.
        """
        pil_img = self.to_pil(frame.image)
        return imagehash.average_hash(pil_img, hash_size)
    
    def dhash_lib(self, frame:Frame, hash_size: int = 8) -> imagehash.ImageHash:
        """
        Compute the difference hash (dHash) of the frame.

        Args and return as in ahash_lib.
        """
        pil_img = self.to_pil(frame.image)
        return imagehash.dhash(pil_img, hash_size)
    
    def phash_lib(self, frame:Frame, hash_size: int = 8) -> imagehash.ImageHash:
        """
        Compute the perceptual hash (pHash) of the frame.

        Args and return as in ahash_lib.
        """
        pil_img = self.to_pil(frame.image)
        return imagehash.phash(pil_img, hash_size)
    
    def md5_lib(self, frame:Frame) -> str:
        """
        Compute the MD5 hash of the raw image bytes.

        Args:
            frame: Frame object containing .image (np.ndarray).

        Returns:
            A hexadecimal MD5 string.
        """
        img_bytes = frame.image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    
    def hamming_distance(self, h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
        """
        Compute the Hamming distance between two perceptual hashes.

        Args:
            h1: First ImageHash.
            h2: Second ImageHash.

        Returns:
            Number of differing bits.
        """
        return h1 - h2
    
    def process(self, frame:Frame) -> Optional[Frame]:
        """
        Filter out near-duplicate frames.

        1. Select the hash function based on self.type_of_hash.
        2. Compute the current frame's hash.
        3. Compare against all stored hashes for this frame.source_frame:
           - For 'md5', drop the frame if the hex strings match.
           - For perceptual hashes, drop if Hamming distance ≤ delta.
        4. If no duplicate found, store the new hash and return the frame.
        5. Otherwise return None.

        Args:
            frame: Incoming Frame to process.

        Returns:
            The same Frame if unique; otherwise None.
        """

        hash_func = self.hash_funcs.get(self.type_of_hash)
        if hash_func is None:
            raise ValueError(f"Unknown hash_type: {self.type_of_hash!r}")
        
        current_hash = hash_func(frame)
        key_folder = frame.source_frame

        for old_hash in self.cache[key_folder].values():
            if self.type_of_hash == 'md5':
                if current_hash == old_hash:
                    return None
            else:
                if self.hamming_distance(current_hash, old_hash) <= self.delta:
                    return None
        
        self.cache[key_folder][frame.frame_id] = current_hash
        return frame
        
