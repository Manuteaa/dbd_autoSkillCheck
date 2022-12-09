
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import Network

if __name__ == '__main__':
    # datasets_path = ["E:/temp/dataset/0", "E:/temp/dataset/0bis"] + ["E:/temp/dataset/1", "E:/temp/dataset/1bis"] * 5
    datasets_path = ["E:/temp/dataset/0bis"] + ["E:/temp/dataset/1", "E:/temp/dataset/1bis"]
    labels_idx = [0] + [1, 1]
    # checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"

    training_data = Network.CustomDataset(datasets_path, labels_idx, transform=Network.transforms)
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    my_model = Network.MyModel()
    # my_model = Network.MyModel.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
    trainer.fit(model=my_model, train_dataloaders=train_dataloader)

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
