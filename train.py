
import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader

import Dbd_Model
import Dbd_DatasetLoader

if __name__ == '__main__':
    ##########################################################
    # checkpoint = None
    checkpoint = "./lightning_logs/version_1/checkpoints/epoch=6-step=5754.ckpt"

    # dataset_0 = ["E:/temp/dbd/0", "E:/temp/dbd/0bis"]
    dataset_0 = ["E:/temp/dbd/0bis"]
    dataset_1 = ["E:/temp/dbd/1", "E:/temp/dbd/1bis"]

    datasets_path = dataset_0 + dataset_1
    labels_idx = [0] + [1, 1]
    ##########################################################

    dataset = Dbd_DatasetLoader.CustomDataset(datasets_path, labels_idx, transform=Dbd_DatasetLoader.transforms)
    training_data, validation_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    training_sampler = Dbd_DatasetLoader.get_randomSampler(training_data.indices)
    training_dataloader = DataLoader(training_data, sampler=training_sampler, batch_size=64)
    validation_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    # test = next(iter(training_dataloader))

    my_model = Dbd_Model.My_Model() if checkpoint is None else Dbd_Model.My_Model.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100)
    trainer.fit(model=my_model, train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
