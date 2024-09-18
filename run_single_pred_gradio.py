import os.path

import gradio as gr

from dbd.AI_model import AI_model


def center_crop(image, crop_size=(224, 224)):
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        gr.Error("Image shape is invalid")

    # Calculate the starting and ending indices for cropping
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    # Ensure the crop indices are valid
    start_y = max(0, start_y)
    start_x = max(0, start_x)

    return image[start_y:start_y + crop_h, start_x:start_x + crop_w, :3]


def predict(onnx_ai_model, image):
    if onnx_ai_model is None or not os.path.exists(onnx_ai_model) or ".onnx" not in onnx_ai_model:
        raise gr.Error("Invalid onnx file", duration=0)

    if image is None:
        raise gr.Error("Invalid image", duration=0)

    # AI model
    ai_model = AI_model(onnx_ai_model)

    image_np = center_crop(image)
    image_np = ai_model.pil_to_numpy(image_np)  # apply transforms, even if input is not a pil image
    pred, probs = ai_model.predict(image_np)

    probs = {ai_model.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}
    return probs


if __name__ == "__main__":
    demo = gr.Interface(fn=predict,
                        inputs=[gr.Dropdown(choices=["model.onnx"], value="model.onnx"), gr.Image()],
                        outputs=gr.Label(label="Skill check recognition")
                        )
    demo.launch()
