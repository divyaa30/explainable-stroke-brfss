import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class MissForestImputer(BaseEstimator, TransformerMixin):
    """MissForest-style imputation (IterativeImputer + random forests).
    Fitted on a stratified subsample of the data passed to fit (the training data);
    columns listed in round_cols (binary / ordinal) are rounded and clipped to their observed range."""

    def __init__(self, n_fit=50000, max_iter=5, round_cols=(), random_state=42):
        self.n_fit = n_fit
        self.max_iter = max_iter
        self.round_cols = round_cols
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(X))
        if len(X) > self.n_fit:
            if y is not None:
                y = np.asarray(y)
                idx = np.concatenate([
                    rng.choice(np.where(y == c)[0], int(round(self.n_fit * (y == c).mean())), replace=False)
                    for c in np.unique(y)])
            else:
                idx = rng.choice(idx, self.n_fit, replace=False)
        forest = RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=5, max_features=0.5,
                                       n_jobs=-1, random_state=self.random_state)
        self.imputer_ = IterativeImputer(estimator=forest, max_iter=self.max_iter, initial_strategy='median',
                                         skip_complete=True, keep_empty_features=True,
                                         random_state=self.random_state).fit(X[idx])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        self.lo_ = np.where(np.isnan(lo), -np.inf, lo)
        self.hi_ = np.where(np.isnan(hi), np.inf, hi)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        Z = self.imputer_.transform(np.asarray(X, dtype=float))
        for j in self.round_cols:
            Z[:, j] = np.clip(np.round(Z[:, j]), self.lo_[j], self.hi_[j])
        return Z