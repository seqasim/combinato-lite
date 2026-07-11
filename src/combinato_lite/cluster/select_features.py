"""Feature selection via KS normality test (ported from Combinato)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ..config import get_config


def select_features(features: np.ndarray) -> np.ndarray:
    cfg = get_config()
    factor = cfg.feature_factor
    num_features_out = cfg.nFeatures
    num_features = features.shape[1]

    feat_std = factor * features.std(0)
    feat_mean = features.mean(0)
    feat_up = feat_mean + feat_std
    feat_down = feat_mean - feat_std

    scores = np.zeros(num_features)
    for i in range(num_features):
        idx = (features[:, i] > feat_down[i]) & (features[:, i] < feat_up[i])
        if idx.any():
            good = features[idx, i]
            good = good - good.mean()
            good /= good.std()
            scores[i] = stats.kstest(good, "norm")[1]

    sorted_scores = np.sort(scores)
    border = sorted_scores[num_features_out]
    ret = (scores <= border).nonzero()[0]
    return ret[:num_features_out]
