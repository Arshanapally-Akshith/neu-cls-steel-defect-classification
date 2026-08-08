"""HOG (Histogram of Oriented Gradients) feature extraction.

Wrapped as an sklearn-compatible transformer so it can sit inside a
Pipeline alongside the scaler and classifier. HOG itself is stateless
(fit is a no-op) — only the downstream StandardScaler and LogisticRegression
have parameters that must be fit on training data only.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog


class HOGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (16, 16),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
    ):
        self.orientations = orientations
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        features = [
            hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                feature_vector=True,
            )
            for image in X
        ]
        return np.vstack(features)
