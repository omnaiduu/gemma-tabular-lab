"""Text-feature reduction that also handles small smoke-test vocabularies."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class AdaptiveSVD(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=16, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        components = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self.reducer_ = TruncatedSVD(n_components=components, random_state=self.random_state).fit(X) if components >= 1 else None
        return self

    def transform(self, X):
        if self.reducer_ is not None:
            return self.reducer_.transform(X)
        return X.toarray() if hasattr(X, "toarray") else X
