import os.path

import gradio as gr
from PIL import Image

from dbd.AI_model import AI_model


def center_crop(image: Image.Image, crop_size=(224, 224)):
    crop_h, crop_w = crop_size
    width, height = image.size

    if height < crop_h or width < crop_w:
        gr.Error("Image shape is invalid")

    left = (width - crop_w) / 2
    top = (height - crop_h) / 2
    right = (width + crop_w) / 2
    bottom = (height + crop_h) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    return image


def predict(onnx_ai_model, image):
    if onnx_ai_model is None or not os.path.exists(onnx_ai_model) or ".onnx" not in onnx_ai_model:
        raise gr.Error("Invalid onnx file", duration=0)

    if image is None:
        raise gr.Error("Invalid image", duration=0)

    # AI model
    ai_model = AI_model(onnx_ai_model)

    image = center_crop(image)
    image = ai_model.pil_to_numpy(image)
    pred, desc, probs, should_hit = ai_model.predict(image)

    return probs


if __name__ == "__main__":
    demo = gr.Interface(fn=predict,
                        inputs=[gr.Dropdown(label="ONNX model filepath", choices=["model.onnx"], value="model.onnx", info="Filepath of the ONNX model (trained AI model)"),
                                gr.Image(type="pil")],
                        outputs=gr.Label(label="Skill check recognition")
                        )
    demo.launch()
