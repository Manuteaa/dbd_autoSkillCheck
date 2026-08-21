from pathlib import Path

import torch

from dbd.networks.model import Model

if __name__ == "__main__":
    checkpoint_dir = Path("lightning_logs/version_7/checkpoints")  # Torch model to export
    onnx_path = "./models/model_v4.onnx"  # destination ONNX model path

    checkpoint = list(checkpoint_dir.glob("*.ckpt"))[-1]
    model = Model.load_from_checkpoint(checkpoint, strict=True).eval()

    # Base ONNX export
    input_sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    model.to_onnx(onnx_path, input_sample, external_data=False)
