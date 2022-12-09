import torch
from torch.utils.data import DataLoader
import Network

if __name__ == '__main__':
    dataset_test = ["E:/temp/dataset/TEST"]
    labels = [1]
    checkpoint = "./lightning_logs/version_2/checkpoints/epoch=9-step=16440.ckpt"

    test_data = Network.CustomDataset(dataset_test, labels, transform=Network.transforms_test)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # my_model = Network.MyModel()
    my_model = Network.MyModel.load_from_checkpoint(checkpoint)

    for sample in test_dataloader.dataset:
        pred = my_model(sample[0].unsqueeze(0))
        y_true = sample[1]
        y_pred = torch.argmax(pred, -1)
        print("pred: {}, true:{}".format(y_pred, y_true))

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
