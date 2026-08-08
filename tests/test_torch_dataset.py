import torch

from src.data.torch_dataset import NEUClsDataset


def test_dataset_reuses_loader_and_matches_manifest(split_manifests, raw_dir, config):
    classes = config["dataset"]["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    sample = split_manifests["train"].head(6)

    dataset = NEUClsDataset(sample, raw_dir, class_to_idx, transform=None)

    assert len(dataset) == len(sample)
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.size == (200, 200)  # PIL Image, no transform applied
        assert label == class_to_idx[sample.iloc[i]["class"]]


def test_dataset_applies_transform(split_manifests, raw_dir, config):
    classes = config["dataset"]["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    sample = split_manifests["train"].head(2)

    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = NEUClsDataset(sample, raw_dir, class_to_idx, transform=transform)

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
