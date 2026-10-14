"""Metrics computed from saved logits (no model needed)."""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, precision_recall_fscore_support


def classification(logits, labels, mask=None, restrict_to=None):
    """Top-1/top-5 accuracy and macro P/R/F1.

    mask: evaluate only these samples. restrict_to: class indices allowed in the
    prediction (e.g. the 101 Food-101 classes for the official-split comparison).
    """
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    if restrict_to is not None:
        allowed = np.zeros(logits.shape[1], bool)
        allowed[list(restrict_to)] = True
        logits = np.where(allowed[None, :], logits, -np.inf)
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    present = np.unique(labels)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, labels=present,
                                                 average="macro", zero_division=0)
    return dict(n=int(len(labels)), acc=float((pred == labels).mean()),
                top5=float((top5 == labels[:, None]).any(1).mean()),
                macro_precision=float(p), macro_recall=float(r), macro_f1=float(f))


def per_class_f1(logits, labels, n_classes):
    pred = logits.argmax(1)
    return f1_score(labels, pred, labels=np.arange(n_classes), average=None, zero_division=0)


def mcnemar(correct_a, correct_b):
    """Chi-square McNemar test with continuity correction (as in the submitted paper)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return dict(b=b, c=c, chi2=float(chi2), p=float(stats.chi2.sf(chi2, 1)))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mean_sd(values):
    v = np.asarray(values, float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0  # sample SD (n-1)
    return float(v.mean()), sd


# ----------------------------------------------------------------- calibration
def fit_temperature(val_logits, val_labels):
    def nll(t):
        return -log_softmax(val_logits / t, axis=1)[np.arange(len(val_labels)), val_labels].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def ece(probs, labels, n_bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def risk_coverage(probs, labels):
    """Selective classification: keep the most confident predictions first."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    order = np.argsort(-conf)
    cum_correct = np.cumsum(correct[order])
    k = np.arange(1, len(labels) + 1)
    coverage = k / len(labels)
    selective_acc = cum_correct / k
    aurc = float(np.mean(1 - selective_acc))
    return coverage, selective_acc, conf[order], aurc


def coverage_at_accuracy(coverage, selective_acc, thresholds, target):
    ok = np.flatnonzero(selective_acc >= target)
    if len(ok) == 0:
        return dict(target=target, coverage=0.0, threshold=None)
    i = ok.max()
    return dict(target=target, coverage=float(coverage[i]), threshold=float(thresholds[i]))


def calibrated_probs(logits, temperature=1.0):
    return softmax(logits / temperature, axis=1)
