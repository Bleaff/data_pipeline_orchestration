"""This module implements the linear Kalman filter in both an object
oriented and procedural form. The KalmanFilter class implements
the filter by storing the various matrices in instance variables,
minimizing the amount of bookkeeping you have to do.
All Kalman filters operate with a predict->update cycle. The
predict step, implemented with the method or function predict(),
uses the state transition matrix F to predict the state in the next
time period (epoch). The state is stored as a gaussian (x, P), where
x is the state (column) vector, and P is its covariance. Covariance
matrix Q specifies the process covariance. In Bayesian terms, this
prediction is called the *prior*, which you can think of colloquially
as the estimate prior to incorporating the measurement.
The update step, implemented with the method or function `update()`,
incorporates the measurement z with covariance R, into the state
estimate (x, P). The class stores the system uncertainty in S,
the innovation (residual between prediction and measurement in
measurement space) in y, and the Kalman gain in k. The procedural
form returns these variables to you. In Bayesian terms this computes
the *posterior* - the estimate after the information from the
measurement is incorporated.
Whether you use the OO form or procedural form is up to you. If
matrices such as H, R, and F are changing each epoch, you'll probably
opt to use the procedural form. If they are unchanging, the OO
form is perhaps easier to use since you won't need to keep track
of these matrices. This is especially useful if you are implementing
banks of filters or comparing various KF designs for performance;
a trivial coding bug could lead to using the wrong sets of matrices.
This module also offers an implementation of the RTS smoother, and
other helper functions, such as log likelihood computations.
The Saver class allows you to easily save the state of the
KalmanFilter class after every update.
"""

import sys
from collections import deque
from copy import deepcopy
from math import exp, log

import numpy as np
from numpy import dot, eye, isscalar, zeros
from scipy.stats import multivariate_normal

from neudc.utils import LOGGER

# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicitly, but perhaps this is clearer
_support_singular = True
try:
    multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    LOGGER.warning(
        "You are using a version of SciPy that does not support the "
        "allow_singular parameter in scipy.stats.multivariate_normal.logpdf(). "
        "Future versions of FilterPy will require a version of SciPy that "
        "implements this keyword",
    )
    _support_singular = False


def reshape_z(z, dim_z, ndim):
    """Ensure z is a (dim_z, 1) shaped vector."""
    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        msg = f"z must be convertible to shape ({dim_z}, 1)"
        raise ValueError(msg)

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


def logpdf(x, mean=None, cov=1, allow_singular=True):
    """Computes the log of the probability density function of the normal
    N(mean, cov) for the data x. The normal may be univariate or multivariate.

    Wrapper for older versions of scipy.multivariate_normal.logpdf which
    don't support support the allow_singular keyword prior to version 0.15.0.

    If it is not supported, and cov is singular or not PSD you may get
    an exception.

    `x` and `mean` may be column vectors, row vectors, or lists.
    """
    flat_mean = np.asarray(mean).flatten() if mean is not None else None

    flat_x = np.asarray(x).flatten()

    if _support_singular:
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
    return multivariate_normal.logpdf(flat_x, flat_mean, cov)


class KalmanFilterXYSR:
    """Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults will
    not give you a functional filter.
    """

    def __init__(self, dim_x, dim_z, dim_u=0, max_obs=50) -> "KalmanFilterXYSR":
        if dim_x < 1:
            msg = "dim_x must be 1 or greater"
            raise ValueError(msg)
        if dim_z < 1:
            msg = "dim_z must be 1 or greater"
            raise ValueError(msg)
        if dim_u < 0:
            msg = "dim_u must be 0 or greater"
            raise ValueError(msg)

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))  # state
        self.P = eye(dim_x)  # uncertainty covariance
        self.Q = eye(dim_x)  # process uncertainty
        self.B = None  # control transition matrix
        self.F = eye(dim_x)  # state transition matrix
        self.H = zeros((dim_z, dim_x))  # measurement function
        self.R = eye(dim_z)  # measurement uncertainty
        self._alpha_sq = 1.0  # fading memory control
        self.M = np.zeros((dim_x, dim_z))  # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # keep all observations
        self.max_obs = max_obs
        self.history_obs = deque([], maxlen=self.max_obs)

        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False
        self.last_measurement = None

    def apply_affine_correction(self, m, t) -> None:
        """Apply to both last state and last observation for OOS smoothing.

        Messy due to internal logic for kalman filter being messy.
        """
        self.x[:2] = m @ self.x[:2] + t
        self.x[4:6] = m @ self.x[4:6]

        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T

        # If frozen, also need to update the frozen state for OOS
        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
            self.attr_saved["x"][4:6] = m @ self.attr_saved["x"][4:6]

            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][4:6, 4:6] = m @ self.attr_saved["P"][4:6, 4:6] @ m.T

            self.attr_saved["last_measurement"][:2] = m @ self.attr_saved["last_measurement"][:2] + t

    def predict(self, u=None, B=None, F=None, Q=None) -> None:
        """Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters.
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.

        """
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def freeze(self) -> None:
        """Save the parameters before non-observation forward."""
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self) -> None:
        if self.attr_saved is not None:
            new_history = deepcopy(list(self.history_obs))
            self.__dict__ = self.attr_saved
            self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1, index2 = indices[-2], indices[-1]
            box1, box2 = new_history[index1], new_history[index2]
            x1, y1, s1, r1 = box1
            w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
            x2, y2, s2, r2 = box2
            w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

            for i in range(index2 - index1):
                x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
                w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
                s, r = w * h, w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                self.update(new_box)
                if i != index2 - index1 - 1:
                    self.predict()
                    self.history_obs.pop()
            self.history_obs.pop()

    def update(self, z, R=None, H=None) -> None:
        """Add a new measurement (z) to the Kalman filter. If z is None, nothing is changed.

        Parameters.
        ----------
        z : np.array
            Measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.
        R : np.array, scalar, or None
            Measurement noise. If None, the filter's self.R value is used.
        H : np.array, or None
            Measurement function. If None, the filter's self.H value is used.

        """
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # append the observation
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                """
                Got no observation so freeze the current parameters for future
                potential online smoothing.
                """
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        # self.observed = True
        if not self.observed:
            """
            Get observation, use online smoothing to re-update parameters
            """
            self.unfreeze()
        self.observed = True

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        self.K = PHT.dot(self.SI)

        # x = x + Ky
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # save history of observations
        self.history_obs.append(z)

    def update_steadystate(self, z, H=None) -> None:
        """Update Kalman filter using the Kalman gain and state covariance
        matrix as computed for the steady state. Only x is updated, and the
        new value is stored in self.x. P is left unchanged. Must be called
        after a prior call to compute_steady_state().
        """
        if z is None:
            self.history_obs.append(z)
            return

        if H is None:
            H = self.H

        H = np.asarray(H)
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # x = x + Ky
        self.x = self.x + dot(self.K_steady_state, self.y)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()

        # save history of observations
        self.history_obs.append(z)

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        return self._log_likelihood

    @log_likelihood.setter
    def log_likelihood(self, z=None):
        """log-likelihood of the measurement z. Computed from the
        system uncertainty S.
        """
        if z is None:
            z = self.z
        return logpdf(z, dot(self.H, self.x), self.S)

    @property
    def likelihood(self):
        """Likelihood of the last measurement."""
        return self._likelihood

    @likelihood.setter
    def likelihood(self, z=None):
        """Likelihood of the measurement z. Computed from the
        system uncertainty S.
        """
        if z is None:
            z = self.z
        return exp(self.log_likelihood(z))
