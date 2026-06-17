from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def single_exp(r, A, l1):
    return A * np.exp(-r / l1)


def double_exp(r, A, l1, l2):
    B = 1.0 - A
    return A * np.exp(-r / l1) + B * np.exp(-r / l2)


def triple_exp(r, A, B, l1, l2, l3):
    C = 1.0 - A - B
    return A * np.exp(-r / l1) + B * np.exp(-r / l2) + C * np.exp(-r / l3)


def gaussian(r, A, l1):
    return A * np.exp(-((r / l1) ** 2))


def damped_cosine(r, A, l1, l2):
    return A * np.exp(-r / l1) * np.cos(2 * np.pi * r / l2)


def stretched_exp(r, A, l1, beta):
    return A * np.exp(-((r / l1) ** beta))


MODELS = {
    "single": (single_exp, ["A", "l1"]),
    "double": (double_exp, ["A", "l1", "l2"]),
    "triple": (triple_exp, ["A", "B", "l1", "l2", "l3"]),
    "gaussian": (gaussian, ["A", "l1"]),
    "damped_cosine": (damped_cosine, ["A", "l1", "l2"]),
    "stretched": (stretched_exp, ["A", "l1", "beta"]),
    "stretched_exp": (stretched_exp, ["A", "l1", "beta"]),
}


# ---------------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    model: str
    popt: np.ndarray
    pcov: np.ndarray
    r2: float
    rmse: float
    rel_err: np.ndarray  # per-parameter relative uncertainty
    param_names: list[str]
    y_pred: np.ndarray

    @property
    def perr(self):
        return np.sqrt(np.diag(self.pcov))

    @property
    def is_reliable(self):
        return self.r2 > 0.99 and np.all(self.rel_err < 0.5)

    def aic(self, n: int) -> float:
        ss_res = np.sum((self.y_pred - self.y_pred) ** 2)  # placeholder
        k = len(self.popt)
        ss_res = self.rmse**2 * n
        return n * np.log(ss_res / n) + 2 * k

    def summary(self, scale: float = 1.0, unit: str = "px") -> str:
        parts = []
        for name, val, err in zip(self.param_names, self.popt, self.perr):
            if name.startswith("l"):
                parts.append(f"{name}={val * scale:.3f}±{err * scale:.3f}{unit}")
            else:
                parts.append(f"{name}={val:.3f}±{err:.3f}")
        parts += [f"R²={self.r2:.4f}", f"RMSE={self.rmse:.4f}"]
        return "  ".join(parts)

    def warning(self) -> str | None:
        bad = [n for n, e in zip(self.param_names, self.rel_err) if e > 0.5]
        if bad:
            return f"Poorly constrained: {bad} - consider a simpler model"
        return None


# ---------------------------------------------------------------------------
# Initialization strategies
# ---------------------------------------------------------------------------
# Each strategy returns a p0 list or None if it cannot produce valid bounds.
# The caller is responsible for checking bounds consistency.
#
# For single/double/triple we define two families of strategies:
#   "positive"  – all amplitudes positive, curve stays >= 0 everywhere
#   "negative"  – at least one amplitude pushed negative so the curve can dip
#
# Roughly 1/4 of starts use the negative family; the rest use positive.
# This ratio is intentionally asymmetric: most autocorrelation curves are
# non-negative, so we do not want to waste the majority of starts on the
# negative region. 1/4 is enough to reliably find a negative-dip solution
# when the data requires it.
# ---------------------------------------------------------------------------

_NEG_FRAC = 0.25  # fraction of starts dedicated to negative-dip initialisation


def _p0_single(rng, x_max, negative, eps):
    """
    Sample p0 for single_exp: [A, l1].

    Positive family : A ∈ (0, 1]  → curve always non-negative.
    Negative family : A > 1        → because B = 1-A < 0 is implicit in the
                                     single model (there is no B), the only
                                     way single_exp itself goes negative is if
                                     A < 0.  We therefore allow A ∈ [-1, 0).
    """
    lo1 = max(eps, x_max * 0.01)
    hi1 = max(lo1 + eps, x_max * 0.5)
    l1  = rng.uniform(lo1, hi1)

    if negative:
        A = rng.uniform(-1.0, -eps)   # negative amplitude → curve dips below 0
    else:
        A = rng.uniform(eps, 1.0)

    # bounds: A is fully free in [-2, 2]; l1 > 0
    bounds = ([-2.0, eps], [2.0, x_max * 2 + eps])
    return [A, l1], bounds


def _p0_double(rng, x_max, negative, eps):
    """
    Sample p0 for double_exp: [A, l1, l2].
    B = 1 - A is implicit.

    Positive family : A ∈ (0, 1) so B ∈ (0, 1) → both terms positive.
    Negative family : A > 1      so B = 1-A < 0 → second (long) term negative.
                   or A < 0      so B = 1-A > 1 → first (short) term negative.
    Both sub-cases are sampled with equal probability within the negative family.
    """
    lo1 = max(eps, x_max * 0.01)
    hi1 = max(lo1 + eps, x_max * 0.1)
    lo2 = hi1
    hi2 = max(lo2 + eps, x_max * 1.0)
    l1  = rng.uniform(lo1, hi1)
    l2  = rng.uniform(lo2, hi2)

    if negative:
        if rng.random() < 0.5:
            A = rng.uniform(1.0 + eps, 2.0)   # B = 1-A < 0  (long arm negative)
        else:
            A = rng.uniform(-1.0, -eps)        # B = 1-A > 1  (short arm negative,
                                               # large positive long arm)
    else:
        A = rng.uniform(eps, 1.0 - eps)

    bounds = ([-2.0, eps, eps], [2.0, x_max * 0.5 + eps, x_max * 2 + eps])
    return [A, l1, l2], bounds


def _p0_triple(rng, x_max, negative, eps):
    """
    Sample p0 for triple_exp: [A, B, l1, l2, l3].
    C = 1 - A - B is implicit.

    Positive family : A > 0, B > 0, A+B < 1  → C > 0, all terms positive.

    Negative family – three sub-cases, each with probability 1/3:
      (a) C < 0  : A+B > 1, both A and B positive.
                   The long (l3) exponential arm is negative.
                   This is the most physically common case (damped oscillation
                   approximated by an overshoot in the slowest component).
      (b) B < 0  : short-mid arm negative.
      (c) A < 0  : short arm negative.
    """
    # tier boundaries
    t1_hi = max(eps * 2, x_max * 0.05)
    t1_lo = max(eps,     t1_hi * 0.1)
    t2_lo = t1_hi
    t2_hi = max(t2_lo + eps, x_max * 0.3)
    t3_lo = max(t2_hi,       x_max * 0.3)
    t3_hi = max(t3_lo + eps, x_max * 2.0)

    l1 = rng.uniform(t1_lo, t1_hi)
    l2 = rng.uniform(t2_lo, t2_hi)
    l3 = rng.uniform(t3_lo, t3_hi)

    if negative:
        sub = rng.integers(3)
        if sub == 0:
            # C < 0: A+B > 1, both positive
            A = rng.uniform(0.5, 1.2)
            B = rng.uniform(0.3, 1.2 - A + 0.4)   # ensures A+B can exceed 1
            B = max(eps, B)
        elif sub == 1:
            # B < 0
            A = rng.uniform(0.5, 1.0)
            B = rng.uniform(-0.8, -eps)
        else:
            # A < 0
            A = rng.uniform(-0.8, -eps)
            B = rng.uniform(eps, 0.8)
    else:
        A = rng.uniform(eps, 0.7)
        B = rng.uniform(eps, min(0.5, 1.0 - A - eps))

    bounds = (
        [-2.0, -2.0, eps, eps, eps],
        [ 2.0,  2.0,
          max(eps, x_max * 0.1),
          max(eps, x_max * 0.5),
          x_max * 2 + eps],
    )
    return [A, B, l1, l2, l3], bounds


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_autocorrelation(
    x: np.ndarray,
    y: np.ndarray,
    model: str = "triple",
    n_starts: int = 50,
    seed: int = 42,
) -> "FitResult | None":
    """
    Robust multi-start fit of an autocorrelation curve.

    For single / double / triple exponential models, roughly 1/4 of random
    starts are initialised in the "negative-dip" region so the optimiser can
    find solutions where one amplitude component is negative and the curve
    drops below zero — as is physically observed in oscillatory or
    anti-correlated signals.

    Parameters
    ----------
    x       : lag axis (pixels or physical units)
    y       : autocorrelation values (need not be normalised to 1)
    model   : "single" | "double" | "triple" | "gaussian" | "damped_cosine"
              | "stretched" | "stretched_exp"
    n_starts: number of random initialisations
    seed    : RNG seed for reproducibility

    Returns
    -------
    FitResult or None if all starts failed
    """
    if not np.any(np.isfinite(y)):
        return None

    func, param_names = MODELS[model]
    rng = np.random.default_rng(seed)
    x_max = float(x.max())
    eps = 1e-3

    best_popt, best_pcov, best_r2 = None, None, -np.inf

    for i in range(n_starts):

        # Decide whether this start should explore negative-dip territory.
        # We use a deterministic schedule (every 4th start) rather than a
        # random coin flip so the coverage is uniform across the run.
        use_negative = (model in ("single", "double", "triple")) and (
            i % round(1.0 / _NEG_FRAC) == 0
        )

        # ------------------------------------------------------------------
        # Build p0 and bounds
        # ------------------------------------------------------------------
        try:
            if model == "single":
                p0, bounds = _p0_single(rng, x_max, use_negative, eps)

            elif model == "double":
                p0, bounds = _p0_double(rng, x_max, use_negative, eps)

            elif model == "triple":
                p0, bounds = _p0_triple(rng, x_max, use_negative, eps)

            elif model == "gaussian":
                lo1 = max(eps, x_max * 0.05)
                hi1 = max(lo1 + eps, x_max * 0.5)
                p0 = [rng.uniform(0.5, 1.0), rng.uniform(lo1, hi1)]
                bounds = ([0, eps], [1, x_max * 2 + eps])

            elif model == "damped_cosine":
                A  = rng.uniform(0.1, 1.0)
                lo1 = max(eps, x_max * 0.05)
                hi1 = max(lo1 + eps, x_max * 0.5)
                l1  = rng.uniform(lo1, hi1)
                l2  = rng.uniform(max(eps, x_max * 0.1), max(eps * 2, x_max))
                p0  = [A, l1, l2]
                bounds = ([0, eps, eps], [1, x_max * 2 + eps, x_max * 4 + eps])

            elif model in ("stretched", "stretched_exp"):
                A   = rng.uniform(0.5, 1.0)
                lo1 = max(eps, x_max * 0.01)
                hi1 = max(lo1 + eps, x_max * 0.5)
                l1  = rng.uniform(lo1, hi1)
                beta = rng.uniform(0.2, 1.5)
                p0   = [A, l1, beta]
                bounds = ([0, eps, 0.1], [1, x_max * 2 + eps, 2.0])

            else:
                raise ValueError(
                    f"Unknown model '{model}'. Available: {sorted(MODELS)}"
                )

        except Exception:
            continue

        # ------------------------------------------------------------------
        # Optimise
        # ------------------------------------------------------------------
        try:
            popt, pcov = curve_fit(
                func,
                x,
                y,
                p0=p0,
                bounds=bounds,
                maxfev=10_000,
            )
        except (RuntimeError, ValueError):
            continue

        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf

        if r2 > best_r2:
            best_r2, best_popt, best_pcov = r2, popt, pcov

    if best_popt is None:
        return None

    y_pred  = func(x, *best_popt)
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    rmse    = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    r2      = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else -np.inf
    perr    = np.sqrt(np.diag(best_pcov))
    rel_err = perr / np.abs(best_popt)

    return FitResult(
        model=model,
        popt=best_popt,
        pcov=best_pcov,
        r2=r2,
        rmse=rmse,
        rel_err=rel_err,
        param_names=param_names,
        y_pred=y_pred,
    )