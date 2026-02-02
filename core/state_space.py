import numpy as np

def _sigmoid_stable(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def _approx_norm_ppf(p: float) -> float:
    """
    Approximate inverse CDF of standard normal.
    (Acklam-like rational approximation)
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow

    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return num / den


class _Kalman1D:
    """
    1D Kalman filter on u_t = log(score_t + eps)

        x_{t+1} = A x_t + w_t,   w_t ~ N(0,Q)
        u_t     = C x_t + v_t,   v_t ~ N(0,R)
    """
    def __init__(self, A=1.0, C=1.0, Q=1e-3, R=1e-2, mu0=0.0, P0=1.0):
        self.A = float(A)
        self.C = float(C)
        self.Q = float(Q)
        self.R = float(R)
        self.mu = float(mu0)
        self.P = float(P0)

    def _predict_state(self):
        mu_pred = self.A * self.mu
        P_pred = (self.A**2) * self.P + self.Q
        return mu_pred, P_pred

    def innovation(self, u_t: float):
        mu_x_pred, P_x_pred = self._predict_state()
        mu_u_pred = self.C * mu_x_pred
        S_t = (self.C**2) * P_x_pred + self.R
        nu = u_t - mu_u_pred
        kappa = (nu**2) / max(S_t, 1e-12)
        return nu, S_t, kappa, mu_u_pred, S_t

    def update(self, u_t: float):
        mu_x_pred, P_x_pred = self._predict_state()
        mu_u_pred = self.C * mu_x_pred
        S_t = (self.C**2) * P_x_pred + self.R
        K = (P_x_pred * self.C) / max(S_t, 1e-12)
        nu = u_t - mu_u_pred
        self.mu = mu_x_pred + K * nu
        self.P = (1.0 - K * self.C) * P_x_pred
        kappa = (nu**2) / max(S_t, 1e-12)
        return {"nu": nu, "S": S_t, "kappa": kappa, "K": K, "mu": self.mu, "P": self.P}

    def predict_obs_next(self):
        """
        Predict u_{t+1} | t : mean and var
        """
        mu_x_pred, P_x_pred = self._predict_state()
        mu_u = self.C * mu_x_pred
        S_u = (self.C**2) * P_x_pred + self.R
        return mu_u, S_u
    def predict_state_ahead(self, k: int):
        # returns (mu_x_{t+k|t}, P_{t+k|t})
        k = int(k)
        if k <= 0:
            return self.mu, self.P

        A = self.A
        Q = self.Q

        # mu_k = A^k * mu
        Ak = A ** k
        mu_k = Ak * self.mu

        # P_k = A^k P (A^k)' + sum_{i=0}^{k-1} A^i Q (A^i)'
        # In 1D: P_k = (A^{2k}) P + Q * sum_{i=0}^{k-1} A^{2i}
        if abs(A) == 1.0:
            sum_term = float(k)  # sum of 1's
        else:
            sum_term = (1.0 - (A ** (2*k))) / (1.0 - (A ** 2))
        P_k = (Ak**2) * self.P + Q * sum_term
        return mu_k, P_k

