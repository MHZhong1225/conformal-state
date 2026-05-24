# methods.py
import os
import numpy as np
import pandas as pd
import copy
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm
import pdb
import warnings

"""
    BASELINES
"""

def trailing_window(
    scores,
    alpha,
    lr, # Dummy argument
    weight_length,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if min(weight_length, t_pred) < np.ceil(1/alpha):
            qs[t] = np.inf
        else:
            qs[t] = np.quantile(scores[max(t_pred-weight_length,0):t_pred], 1-alpha, method='higher')
    results = {"method": "Trail", "q" : qs}
    return results

def aci_clipped(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.inf
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
        if qs[t] == np.inf:
            qs[t] = clip_value
    results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas}
    return results


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return results

"""
    New methods
"""

def quantile(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    T_burnin = kwargs['T_burnin']
    results = quantile_integrator_log(scores, alpha, lr, 1.0, 0, ahead, T_burnin, proportional_lr=proportional_lr)
    results['method'] = 'Quantile'
    return results

def mytan(x):
    if x >= np.pi/2:
        return np.inf
    elif x <= -np.pi/2:
        return -np.inf
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

def quantile_integrator_log(
    scores,
    alpha,
    lr,
    Csat,
    KI,
    ahead,
    T_burnin,
    proportional_lr=True,
    *args,
    **kwargs
):
    data = kwargs['data'] if 'data' in kwargs.keys() else None
    results = quantile_integrator_log_scorecaster(scores, alpha, lr, data, T_burnin, Csat, KI, True, ahead, proportional_lr=proportional_lr, scorecast=False)
    results['method'] = "Quantile+Integrator (log)"
    return results


"""
    This is the master method for the quantile, integrator, and scorecaster methods.
"""
def quantile_integrator_log_scorecaster(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
#    onesided_integrator=False,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1
    # Load the scorecaster
    try:
        # If the data contains a scorecasts column, then use it!
        if 'scorecasts' in data.columns:
            scorecasts = np.array([s[int(upper)] for s in data['scorecasts'] ])
            train_model = False
        else:
            scorecasts = np.load('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy')
            train_model = False
    except:
        train_model = True
    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    #lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)
        #integrator = saturation_fn_log((1-covereds)[T_burnin:t_pred].sum() - (t_pred-T_burnin)*alpha, (t_pred-T_burnin), Csat, KI) if t_pred > T_burnin else 0
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        #if onesided_integrator:
        #    integrator_arg = np.clip(integrator_arg, 0, np.inf)
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        # Train and scorecast if necessary
        if scorecast and train_model and t_pred > T_burnin and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[:t_pred])
            model = ThetaModel(
                    curr_scores.astype(float),
                    period=seasonal_period,
                    ).fit()
            pred = model.forecast(ahead)          # Series, len=ahead
            scorecasts[t+ahead] = float(np.asarray(pred)[-1])
        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs}
    if train_model and scorecast:
        os.makedirs('./.cache/', exist_ok=True)
        os.makedirs('./.cache/scorecaster/', exist_ok=True)
        np.save('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy', scorecasts)
    return results


from tqdm import tqdm
import numpy as np
from .state_space import _Kalman1D, _approx_norm_ppf, _sigmoid_stable


def dss_cc(
    scores,
    alpha,
    lr,    
    ahead,
    T_burnin,

    q_max=10.0,
    eps=1e-6,

    # gain scheduling
    eta_min=0.01,
    eta_max=0.2,
    kappa_center=3.0,
    kappa_scale=1.0,

    # kalman params
    kf_A=1.0,
    kf_C=1.0,
    kf_Q=1e-3,
    kf_R=1e-2,

    # feedforward construction
    use_var_term=True,
    z_value=None,

    proportional_eta=True,  
    *args,
    **kwargs
):
    scores = np.asarray(scores, dtype=float)
    T = scores.shape[0]
    horizon = max(int(ahead), 1)

    q_max = float(q_max)
    eps = float(eps)

    eta_min = float(eta_min)
    eta_max = float(eta_max)
    kappa_center = float(kappa_center)
    kappa_scale = float(kappa_scale)

    def clip_q(x):
        return float(np.clip(x, 0.0, q_max))

    def schedule_eta(kappa):
        x = (kappa - kappa_center) / max(kappa_scale, 1e-12)
        psi = _sigmoid_stable(x)
        return eta_min + (eta_max - eta_min) * psi

    def empirical_quantile(end):
        history = scores[:max(int(end), 0)]
        history = history[np.isfinite(history)]
        if history.size == 0:
            return q_max
        if history.size < min_needed:
            return history.max()
        return np.quantile(history, 1.0 - alpha, method="higher")

    def recent_score_scale(end):
        if not proportional_eta or end <= 0:
            return 1.0
        start = max(end - T_burnin, 0)
        window = scores[start:end]
        window = window[np.isfinite(window)]
        if window.size <= 1:
            return 1.0
        return max(float(window.max() - window.min()), 1e-12)

    def predict_feedforward():
        mu_x_pred, P_x_pred = kf.predict_state_ahead(horizon)
        mu_u_pred = kf.C * mu_x_pred
        S_u_pred = (kf.C**2) * P_x_pred + kf.R
        if use_var_term:
            log_q = mu_u_pred + z_quant * np.sqrt(max(S_u_pred, 1e-12))
        else:
            log_q = mu_u_pred
        return float(np.exp(log_q))

    z_quant = float(_approx_norm_ppf(1.0 - alpha)) if z_value is None else float(z_value)

    # outputs
    qs = np.zeros((T,), dtype=float)        # q[t] used for coverage at time t
    z_fb = np.zeros((T,), dtype=float)      # feedback state
    g_ff = np.zeros((T,), dtype=float)      # feedforward used at time t
    covereds = np.zeros((T,), dtype=float)
    etas = np.zeros((T,), dtype=float)
    kappas = np.zeros((T,), dtype=float)

    # kalman on u_t = log(score_t + eps)
    kf = _Kalman1D(A=kf_A, C=kf_C, Q=kf_Q, R=kf_R, mu0=0.0, P0=1.0)
    use_residual_feedback = np.nanmin(scores) >= 0.0

    # cold start threshold (like other methods: need enough samples)
    min_needed = int(np.ceil(1.0 / alpha))

    for t in tqdm(range(T)):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue

        # 1) coverage uses qs[t] exactly like PID baseline
        covereds[t] = 1.0 if (qs[t] >= scores[t]) else 0.0

        # 2) kalman innovation and update using current u_t
        s = float(scores[t])
        u = float(np.log(max(s, 0.0) + eps))

        nu, S_t, kappa, _, _ = kf.innovation(u)
        kappas[t] = float(kappa)

        base_eta = schedule_eta(kappa)   # in [eta_min, eta_max]

        # PID-like scaling by recent score range, matching the existing tuning.
        eta_t = lr * base_eta * recent_score_scale(t)
        etas[t] = float(eta_t)

        kf.update(u)

        # 3) burn-in: use the empirical score scale instead of jumping to q_max.
        if t < T - 1:
            if t_pred < min_needed:
                g_ff[t+1] = clip_q(empirical_quantile(t_pred))
                qs[t+1] = g_ff[t+1]
                continue

        # 4) compute feedforward for the configured forecast horizon.
        g_next = predict_feedforward()

        # 5) feedback controls residuals for nonnegative scores. Signed residual
        # scores can be negative, so they keep the original coverage feedback.
        if use_residual_feedback:
            update_hit = 1.0 if (scores[t_pred] - g_ff[t_pred] <= z_fb[t_pred]) else 0.0
        else:
            update_hit = covereds[t_pred]
        grad = alpha if (update_hit > 0.5) else -(1.0 - alpha)

        if t < T - 1:
            z_fb[t+1] = z_fb[t] - eta_t * grad
            g_ff[t+1] = clip_q(g_next)
            qs[t+1] = clip_q(z_fb[t+1] + g_ff[t+1])
        # if t % 200 == 0 and t > 0:
        #     print("t", t, "q_mean", qs[max(0,t-200):t].mean(), "q_min", qs[:t].min(), "q_max", qs[:t].max(),
        #         "cvg_recent", covereds[max(0,t-200):t].mean(), "eta", etas[t])
    return {
        "method": "DSS-CC",
        "q": qs,
        "eta": etas,
        "kappa": kappas,
        "z": z_fb,
        "g": g_ff,
        "covered": covereds,
    }


def cptc(
    scores,
    alpha,
    lr,
    z_probs=None,
    *args,
    **kwargs
):
    """
    Conformal Prediction for Time-series with Change points (CPTC)
    Reference: https://arxiv.org/abs/2509.02844
    
    Args:
        scores: (T,) non-conformity scores (e.g. absolute residuals)
        alpha: target miscoverage rate (e.g. 0.1 for 90% coverage)
        lr: learning rate (gamma) for online update
        z_probs: (T, K) probability of each latent state at each time step
    """
    scores = np.asarray(scores, dtype=float)
    
    if z_probs is None:
        # If not provided, assume single state (equivalent to ACI)
        # Or raise error. Given usage in base_test.py, we might want to default.
        # But base_test.py injects it into kwargs.
        # If it comes from kwargs, it will be in z_probs argument due to **kwargs unpacking?
        # No, if it's a named argument in the function, **kwargs in the call will populate it.
        # But if base_test.py puts it in kwargs, and I define it as named arg, it works.
        # If base_test.py DOES NOT put it (e.g. direct call), we handle it here.
        if 'z_probs' in kwargs:
             z_probs = kwargs['z_probs']
        else:
             # Default to single state
             T = scores.shape[0]
             z_probs = np.ones((T, 1))

    z_probs = np.asarray(z_probs, dtype=float)
    T = scores.shape[0]
    K = z_probs.shape[1]

    # Check alignment
    if z_probs.shape[0] != T:
        # If lengths differ, try to truncate or complain. 
        # Assuming aligned for now.
        pass

    # Initialize state-specific quantiles
    # We start with 0.0. 
    # To speed up convergence, one could use a burn-in or heuristic, 
    # but 0.0 is the standard "uninformed" start for ACI.
    qs_states = np.zeros(K, dtype=float)
    
    # Outputs
    qs = np.zeros(T, dtype=float)
    covereds = np.zeros(T, dtype=float)
    
    # We can also track the weighted update norm or something if useful
    
    for t in tqdm(range(T)):
        # 1. Form prediction for current step t using CURRENT state estimate
        # (Assuming z_probs[t] is the predicted state distribution for time t)
        current_q = np.dot(z_probs[t], qs_states)
        qs[t] = current_q
        
        # 2. Check coverage
        s = scores[t]
        covered = 1.0 if s <= current_q else 0.0
        covereds[t] = covered
        
        # 3. Online Update (ACI-style, weighted by state responsibility)
        # Gradient: if not covered (s > q), we need to INCREASE q. 
        # Target quantile is 1-alpha.
        # Quantile regression gradient direction:
        # If y <= q (covered): grad ~ -(1-alpha) -> decrease q
        # If y > q (not covered): grad ~ alpha -> increase q 
        # WAIT. ACI rule is: q <- q + lr * (err - alpha)
        # err = 0 if covered, 1 if not.
        # If covered: err=0 => q <- q - lr * alpha (Decrease)
        # If not covered: err=1 => q <- q + lr * (1 - alpha) (Increase)
        
        err = 0.0 if covered > 0.5 else 1.0
        update = lr * (err - alpha)
        
        # Distribute update to states based on their probability
        # qs_states[k] += z_probs[t, k] * update
        qs_states += z_probs[t] * update
        
        # Clip to ensure non-negative width (since scores are absolute errors)
        qs_states = np.maximum(qs_states, 0.0)

    return {
        "method": "CPTC",
        "q": qs,
        "covered": covereds,
        "qs_states": qs_states # Final state
    }
