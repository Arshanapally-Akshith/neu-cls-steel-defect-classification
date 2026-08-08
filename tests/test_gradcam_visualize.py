import numpy as np

from src.gradcam.visualize import plot_image_grid


def test_plot_image_grid_saves_file(tmp_path):
    images = [np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8) for _ in range(5)]
    titles = [f"img {i}" for i in range(5)]
    save_path = tmp_path / "grid.png"

    plot_image_grid(images, titles, save_path, n_cols=3)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_image_grid_noop_on_empty_list(tmp_path):
    save_path = tmp_path / "empty.png"
    plot_image_grid([], [], save_path)
    assert not save_path.exists()


def test_plot_image_grid_single_image(tmp_path):
    images = [np.zeros((10, 10, 3), dtype=np.uint8)]
    save_path = tmp_path / "single.png"
    plot_image_grid(images, ["only"], save_path, n_cols=3)
    assert save_path.exists()
