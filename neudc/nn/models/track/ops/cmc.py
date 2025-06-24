import copy
from abc import ABC, abstractmethod

import cv2
import numpy as np

from neudc.utils import LOGGER

__all__ = "ECC", "ORB", "SOF", "SIFT"


class BaseCMC(ABC):

    @abstractmethod
    def apply(self, im):
        pass

    def generate_mask(self, img, dets, scale):
        h, w = img.shape
        mask = np.zeros_like(img)

        mask[int(0.02 * h) : int(0.98 * h), int(0.02 * w) : int(0.98 * w)] = 255
        if dets is not None:
            for det in dets:
                tlbr = np.multiply(det, scale).astype(int)
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0

        return mask

    def preprocess(self, img):

        # bgr2gray
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize
        if self.scale is not None:
            img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

        return img


class SIFT(BaseCMC):

    def __init__(
        self,
        warp_mode=cv2.MOTION_EUCLIDEAN,
        eps=1e-5,
        max_iter=100,
        scale=0.1,
        grayscale=True,
        draw_keypoint_matches=False,
        align=False,
    ) -> None:
        """Compute the warp matrix from src to dst.

        Parameters
        ----------
        warp_mode: opencv flag
            translation: cv2.MOTION_TRANSLATION
            rotated and shifted: cv2.MOTION_EUCLIDEAN
            affine(shift,rotated,shear): cv2.MOTION_AFFINE
            homography(3d): cv2.MOTION_HOMOGRAPHY
        eps: float
            the threshold of the increment in the correlation coefficient between two iterations
        max_iter: int
            the number of iterations.
        scale: float or [int, int]
            scale_ratio: float
            scale_size: [W, H]
        align: bool
            whether to warp affine or perspective transforms to the source image
        grayscale: bool
            whether to transform 3 channel RGB to single channel grayscale for faster computations

        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray

        """
        self.grayscale = grayscale
        self.scale = scale
        self.warp_mode = warp_mode
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            self.warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)

        self.detector = cv2.SIFT_create(nOctaveLayers=2, contrastThreshold=0.5, edgeThreshold=10)
        self.extractor = cv2.SIFT_create(nOctaveLayers=2, contrastThreshold=0.5, edgeThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.prev_img = None
        self.minimum_features = 10
        self.prev_desc = None
        self.draw_keypoint_matches = draw_keypoint_matches
        self.align = align

    def apply(self, img: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Apply ORB-based sparse optical flow to compute the warp matrix.

        Parameters
        ----------
        img : ndarray
            The input image.
        dets : ndarray
            Detected bounding boxes in the image.

        Returns
        -------
        ndarray
            The warp matrix from the matching keypoint in the previous image to the current.
            The warp matrix is always 2x3.

        """
        H = np.eye(2, 3)

        img = self.preprocess(img)
        h, w = img.shape

        # generate dynamic object mask
        mask = self.generate_mask(img, dets, self.scale)

        # find static keypoints
        keypoints = self.detector.detect(img, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(img, keypoints)

        # handle first frame
        if self.prev_img is None:
            # Initialize data
            self.prev_dets = dets.copy()
            self.prev_img = img.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prev_img = img.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)

            return H

        # filtered matches based on smallest spatial distance
        matches = []
        spatial_distances = []
        max_spatial_distance = 0.25 * np.array([w, h])

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prev_keypoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatial_distance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )

                if (np.abs(spatial_distance[0]) < max_spatial_distance[0]) and (
                    np.abs(spatial_distance[1]) < max_spatial_distance[1]
                ):
                    spatial_distances.append(spatial_distance)
                    matches.append(m)

        mean_spatial_distances = np.mean(spatial_distances, 0)
        std_spatial_distances = np.std(spatial_distances, 0)

        inliers = (spatial_distances - mean_spatial_distances) < 2.5 * std_spatial_distances

        good_matches = [matches[i] for i in range(len(matches)) if np.all(inliers[i])]

        prevPoints = np.array([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        currPoints = np.array([keypoints[m.trainIdx].pt for m in good_matches])

        # Draw the keypoint matches on the output image
        if self.draw_keypoint_matches:
            self.prev_img[:, :][mask == True] = 0  # noqa:E712
            self.matches_img = np.hstack((self.prev_img, img))
            self.matches_img = cv2.cvtColor(self.matches_img, cv2.COLOR_GRAY2BGR)

            W = np.size(self.prev_img, 1)
            for m in good_matches:
                prev_pt = np.array(self.prev_keypoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))
                self.matches_img = cv2.line(self.matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                self.matches_img = cv2.circle(self.matches_img, prev_pt, 2, tuple(color), -1)
                self.matches_img = cv2.circle(self.matches_img, curr_pt, 2, tuple(color), -1)
            for det in dets:
                det = np.multiply(det, self.scale).astype(int)
                start = (det[0] + w, det[1])
                end = (det[2] + w, det[3])
                self.matches_img = cv2.rectangle(self.matches_img, start, end, (0, 0, 255), 2)
            for det in self.prev_dets:
                det = np.multiply(det, self.scale).astype(int)
                start = (det[0], det[1])
                end = (det[2], det[3])
                self.matches_img = cv2.rectangle(self.matches_img, start, end, (0, 0, 255), 2)
        else:
            self.matches_img = None

        # find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # upscale warp matrix to original images size
            if self.scale < 1.0:
                H[0, 2] /= self.scale
                H[1, 2] /= self.scale

            if self.align:
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, H, (w, h), flags=cv2.INTER_LINEAR)
        else:
            pass

        # Store to next iteration
        self.prev_img = img.copy()
        self.prev_keypoints = copy.copy(keypoints)
        self.prev_descriptors = copy.copy(descriptors)

        return H


class ECC(BaseCMC):
    def __init__(
        self,
        warp_mode: int = cv2.MOTION_EUCLIDEAN,
        eps: float = 1e-5,
        max_iter: int = 100,
        scale: float = 0.1,
        align: bool = False,
        grayscale: bool = True,
    ) -> None:
        """Compute the warp matrix from src to dst.

        Parameters
        ----------
        warp_mode: opencv flag
            translation: cv2.MOTION_TRANSLATION
            rotated and shifted: cv2.MOTION_EUCLIDEAN
            affine(shift,rotated,shear): cv2.MOTION_AFFINE
            homography(3d): cv2.MOTION_HOMOGRAPHY
        eps: float
            the threshold of the increment in the correlation coefficient between two iterations
        max_iter: int
            the number of iterations.
        scale: float or [int, int]
            scale_ratio: float
            scale_size: [W, H]
        align: bool
            whether to warp affine or perspective transforms to the source image
        grayscale: bool
            whether to transform 3 channel RGB to single channel grayscale for faster computations

        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray

        """
        self.align = align
        self.grayscale = grayscale
        self.scale = scale
        self.warp_mode = warp_mode
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        self.prev_img = None

    def apply(self, img: np.ndarray, dets: np.ndarray = None) -> np.ndarray:
        """Apply sparse optical flow to compute the warp matrix.

        Parameters
        ----------
            img (ndarray): The input image.
            dets: Description of dets parameter.

        Returns
        -------
            ndarray: The warp matrix from the source to the destination.
                If the motion model is homography, the warp matrix will be 3x3; otherwise, it will be 2x3.

        """
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        if self.prev_img is None:
            self.prev_img = self.preprocess(img)
            return warp_matrix

        img = self.preprocess(img)

        try:
            (ret_val, warp_matrix) = cv2.findTransformECC(
                self.prev_img,
                img,
                warp_matrix,
                self.warp_mode,
                self.termination_criteria,
                None,
                1,  # already processed
            )
        except Exception as e:
            LOGGER.warning(f"Affine matrix could not be generated: {e}. Returning identity")
            return warp_matrix

        # upscale warp matrix to original images size
        if self.scale < 1:
            warp_matrix[0, 2] /= self.scale
            warp_matrix[1, 2] /= self.scale

        if self.align:
            h, w = self.prev_img.shape
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                self.prev_img_aligned = cv2.warpPerspective(self.prev_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
        else:
            self.prev_img_aligned = None

        self.prev_img = img

        return warp_matrix  # , prev_img_aligned


class ORB(BaseCMC):

    def __init__(
        self,
        feature_detector_threshold: int = 20,
        matcher_norm_type: int = cv2.NORM_HAMMING,
        scale: float = 0.1,
        grayscale: bool = True,
        draw_keypoint_matches: bool = False,
        align: bool = False,
    ) -> None:
        """Compute the warp matrix from src to dst.

        Parameters
        ----------
        feature_detector_threshold: int, optional
            The threshold for feature extraction. Defaults to 20.
        matcher_norm_type: int, optional
            The norm type of the matcher. Defaults to cv2.NORM_HAMMING.
        scale: float, optional
            Scale ratio. Defaults to 0.1.
        grayscale: bool, optional
            Whether to transform 3-channel RGB to single-channel grayscale for faster computations.
            Defaults to True.
        draw_keypoint_matches: bool, optional
            Whether to draw keypoint matches on the output image. Defaults to False.
        align: bool, optional
            Whether to align the images based on keypoint matches. Defaults to False.

        """
        self.grayscale = grayscale
        self.scale = scale

        self.detector = cv2.FastFeatureDetector_create(threshold=feature_detector_threshold)
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(matcher_norm_type)

        self.prev_img = None
        self.draw_keypoint_matches = draw_keypoint_matches
        self.align = align

    def apply(self, img: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Apply ORB-based sparse optical flow to compute the warp matrix.

        Parameters
        ----------
        img : ndarray
            The input image.
        dets : ndarray
            Detected bounding boxes in the image.

        Returns
        -------
        ndarray
            The warp matrix from the matching keypoint in the previous image to the current.
            The warp matrix is always 2x3.

        """
        H = np.eye(2, 3)

        img = self.preprocess(img)
        h, w = img.shape

        # generate dynamic object mask
        mask = self.generate_mask(img, dets, self.scale)

        # find static keypoints
        keypoints = self.detector.detect(img, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(img, keypoints)

        # handle first frame
        if self.prev_img is None:
            # Initialize data
            self.prev_dets = dets.copy()
            self.prev_img = img.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prev_img = img.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)

            return H

        # filtered matches based on smallest spatial distance
        matches = []
        spatial_distances = []
        max_spatial_distance = 0.25 * np.array([w, h])

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prev_keypoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatial_distance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )

                if (np.abs(spatial_distance[0]) < max_spatial_distance[0]) and (
                    np.abs(spatial_distance[1]) < max_spatial_distance[1]
                ):
                    spatial_distances.append(spatial_distance)
                    matches.append(m)

        mean_spatial_distances = np.mean(spatial_distances, 0)
        std_spatial_distances = np.std(spatial_distances, 0)

        inliesrs = (spatial_distances - mean_spatial_distances) < 2.5 * std_spatial_distances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prev_keypoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # draw keypoint matches on the output image
        if self.draw_keypoint_matches:
            self.prev_img[:, :][mask == True] = 0  # noqa:E712
            self.matches_img = np.hstack((self.prev_img, img))
            self.matches_img = cv2.cvtColor(self.matches_img, cv2.COLOR_GRAY2BGR)

            W = np.size(self.prev_img, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prev_keypoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))
                self.matches_img = cv2.line(self.matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                self.matches_img = cv2.circle(self.matches_img, prev_pt, 2, tuple(color), -1)
                self.matches_img = cv2.circle(self.matches_img, curr_pt, 2, tuple(color), -1)
            for det in dets:
                det = np.multiply(det, self.scale).astype(int)
                start = (det[0] + w, det[1])
                end = (det[2] + w, det[3])
                self.matches_img = cv2.rectangle(self.matches_img, start, end, (0, 0, 255), 2)
            for det in self.prev_dets:
                det = np.multiply(det, self.scale).astype(int)
                start = (det[0], det[1])
                end = (det[2], det[3])
                self.matches_img = cv2.rectangle(self.matches_img, start, end, (0, 0, 255), 2)
        else:
            self.matches_img = None

        # find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # upscale warp matrix to original images size
            if self.scale < 1.0:
                H[0, 2] /= self.scale
                H[1, 2] /= self.scale

            if self.align:
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, H, (w, h), flags=cv2.INTER_LINEAR)
        else:
            pass

        # Store to next iteration
        self.prev_img = img.copy()
        self.prev_keypoints = copy.copy(keypoints)
        self.prev_descriptors = copy.copy(descriptors)

        return H


class SOF(BaseCMC):

    def __init__(
        self,
        warp_mode=cv2.MOTION_EUCLIDEAN,
        eps=1e-5,
        max_iter=100,
        scale=0.1,
        align=False,
        grayscale=True,
        draw_optical_flow=False,
    ) -> None:
        """Compute the warp matrix from src to dst.

        Parameters
        ----------
        warp_mode: opencv flag
            translation: cv2.MOTION_TRANSLATION
            rotated and shifted: cv2.MOTION_EUCLIDEAN
            affine(shift,rotated,shear): cv2.MOTION_AFFINE
            homography(3d): cv2.MOTION_HOMOGRAPHY
        eps: float
            the threshold of the increment in the correlation coefficient between two iterations
        max_iter: int
            the number of iterations.
        scale: float or [int, int]
            scale_ratio: float
            scale_size: [W, H]
        align: bool
            whether to warp affine or perspective transforms to the source image
        grayscale: bool
            whether to transform 3 channel RGB to single channel grayscale for faster computations

        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray

        """
        self.align = align
        self.grayscale = grayscale
        self.scale = scale
        self.prev_img = None
        self.draw_optical_flow = draw_optical_flow

        self.detector = cv2.FastFeatureDetector_create(threshold=20)
        self.extractor = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def apply(self, img: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Apply ORB-based sparse optical flow to compute the warp matrix.

        Parameters
        ----------
        img : ndarray
            The input image.
        dets : ndarray
            Detected bounding boxes in the image.

        Returns
        -------
        ndarray
            The warp matrix from the matching keypoint in the previous image to the current.
            The warp matrix is always 2x3.

        """
        H = np.eye(2, 3)

        img = self.preprocess(img)

        h, w = img.shape

        # Lucas-Kanade is based on a local motion constancy assumption,
        # where nearby pixels have the same displacement direction. Hence, it is better to discard
        # dynamic object for feature extraction from static objects
        mask = self.generate_mask(img, dets, self.scale)

        # handle first frame
        if self.prev_img is None:

            # find keypoints in first frame
            keypoints = cv2.goodFeaturesToTrack(
                img,
                mask=mask,
                maxCorners=3000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )

            # if image lacks distinctive features, ignore this frame and try to initialize again
            if keypoints is None:
                return H
            # initialize first frame
            else:
                # Initialize data
                self.prev_img = img.copy()
                self.prev_keypoints = keypoints.copy()
                return H

        # sparse otical flow for sparse features using Lucas-Kanade with pyramids
        # calculate new positions of the keypoints between the previous frame (self.prev_img)
        # and the current frame (img) using sparse optical flow (Lucas-Kanade with pyramids)
        try:
            next_keypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img, self.prev_keypoints, None)
        except Exception as e:
            LOGGER.warning(f"calcOpticalFlowPyrLK failed: {e}")
            return H

        # for simplicity, if no keypoints are found, we discard the frame
        if next_keypoints is None:
            return H

        # keep points that were successfully matched
        self.prev_keypoints = self.prev_keypoints[status == 1].reshape(-1, 1, 2)
        next_keypoints = next_keypoints[status == 1].reshape(-1, 1, 2)

        # get affine matrix
        try:
            H, _ = cv2.estimateAffinePartial2D(self.prev_keypoints, next_keypoints, cv2.RANSAC)
        except Exception as e:
            LOGGER.warning(f"Affine matrix could not be generated: {e}")
            return H
        finally:
            if H is None:
                return np.eye(2, 3)

        if self.draw_optical_flow:
            self.warped_img = cv2.warpAffine(self.prev_img, H, (w, h), flags=cv2.INTER_LINEAR)
            self.mask = np.zeros_like(img)
            for _i, (new, old) in enumerate(zip(next_keypoints, self.prev_keypoints)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(
                    img=self.mask,
                    pt1=tuple(np.int32([a, b])),
                    pt2=tuple(np.int32([c, d])),
                    color=(255, 255, 255),
                    thickness=1,
                )
                self.mask = cv2.circle(
                    img=self.mask,
                    center=tuple(np.int32([a, b])),
                    radius=1,
                    color=(255, 255, 255),
                    thickness=2,
                )

        # store to next iteration
        self.prev_img = img.copy()
        self.prevKeyPoints = next_keypoints.copy()

        # handle downscale
        if self.scale < 1:
            H[0, 2] /= self.scale
            H[1, 2] /= self.scale

        return H
