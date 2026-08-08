import numpy as np
import pytest

from src.features.hog import HOGFeatureExtractor

HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")


@pytest.fixture
def synthetic_images():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(200, 200), dtype=np.uint8) for _ in range(5)]


def test_fit_is_noop_and_returns_self(synthetic_images):
    extractor = HOGFeatureExtractor(**HOG_PARAMS)
    result = extractor.fit(synthetic_images)
    assert result is extractor


def test_transform_output_shape(synthetic_images):
    extractor = HOGFeatureExtractor(**HOG_PARAMS)
    features = extractor.transform(synthetic_images)
    assert features.shape[0] == len(synthetic_images)
    assert features.ndim == 2
    assert features.shape[1] > 0


def test_transform_is_deterministic(synthetic_images):
    extractor = HOGFeatureExtractor(**HOG_PARAMS)
    f1 = extractor.transform(synthetic_images)
    f2 = extractor.transform(synthetic_images)
    np.testing.assert_array_equal(f1, f2)


def test_transform_does_not_mutate_input(synthetic_images):
    before = [img.copy() for img in synthetic_images]
    extractor = HOGFeatureExtractor(**HOG_PARAMS)
    extractor.transform(synthetic_images)
    for orig, after in zip(before, synthetic_images):
        np.testing.assert_array_equal(orig, after)


def test_different_images_yield_different_features():
    rng = np.random.default_rng(1)
    img_a = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
    img_b = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
    extractor = HOGFeatureExtractor(**HOG_PARAMS)
    features = extractor.transform([img_a, img_b])
    assert not np.allclose(features[0], features[1])


def test_feature_dim_matches_hog_math_for_default_params():
    # 200x200 image, pixels_per_cell=16 -> 12x12 cells (floor(200/16)),
    # cells_per_block=2 -> 11x11 block positions, orientations=9.
    extractor = HOGFeatureExtractor(**HOG_PARAMS)
    img = np.zeros((200, 200), dtype=np.uint8)
    features = extractor.transform([img])
    expected_dim = 11 * 11 * 2 * 2 * 9
    assert features.shape[1] == expected_dim
