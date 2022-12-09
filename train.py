
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import Network

if __name__ == '__main__':
    ##########################################################
    checkpoint = None
    # checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"

    # dataset_0 = ["E:/temp/dbd/0", "E:/temp/dbd/0bis"]
    dataset_0 = ["E:/temp/dbd/0bis"]
    dataset_1 = ["E:/temp/dbd/1", "E:/temp/dbd/1bis"]
    datasets_path = dataset_0 + dataset_1
    labels_idx = [0] + [1, 1]
    ##########################################################

    training_data = Network.CustomDataset(datasets_path, labels_idx, transform=Network.transforms)
    sampler = training_data.getSampler()
    train_dataloader = DataLoader(training_data, sampler=sampler, batch_size=64)
    # test = next(iter(train_dataloader))

    my_model = Network.MyModel() if checkpoint is None else Network.MyModel.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100)
    trainer.fit(model=my_model, train_dataloaders=train_dataloader)

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
