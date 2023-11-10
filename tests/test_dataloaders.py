import dbd.datasets.datasetLoader as datasetLoader


dataset_root = "tests/data"

def test_dataset_parsing():
    dataset = datasetLoader._parse_dbd_datasetfolder(dataset_root)
    assert len(dataset) > 0
    assert dataset.shape[1] == 2


def test_get_dataloaders():
    dataloader_train, dataloader_val = datasetLoader.get_dataloaders(dataset_root, batch_size=2, num_workers=1)
    assert len(dataloader_train) > 0
    assert len(dataloader_val) > 0

    batch_train = next(iter(dataloader_train))
    assert len(batch_train) == 2

    x, y = batch_train
    assert x.ndim == 4
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == 2

