import torch
import numpy as np
import gradio as gr
from PIL import Image

class IMDModel(torch.nn.Module):
    def __init__(self):
        super(IMDModel, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.down_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            self.maxpool,
            self.relu
        )
        self.down_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            self.maxpool,
            self.relu
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=16 * 30 * 30, out_features=1024),
            torch.nn.BatchNorm1d(1024),
            self.relu,
            torch.nn.Linear(in_features=1024, out_features=64),
            torch.nn.BatchNorm1d(64),
            self.relu,
            torch.nn.Linear(in_features=64, out_features=2),
            torch.nn.Softmax()
        )

    def forward(self, img):
        d1 = self.down_conv1(img)
        d2 = self.down_conv2(d1)
        d2 = d2.view(-1, d2.shape[1] * d2.shape[2] * d2.shape[3])
        out = self.linear(d2)
        return out

def infer(img_path):
    print("Performing Level 1 analysis...")
    # findMetadata(img_path=img_path)
    #
    # print("Performing Level 2 analysis...")
    # ELA(img_path=img_path)

    img = Image.open(img_path)
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    img = np.expand_dims(img, axis=0)

    model_path = "model/model_c1.pth"
    model = torch.load(model_path, map_location=torch.device('cpu'))
    out = model(torch.from_numpy(img))
    y_pred = torch.max(out, dim=1)[1]

    return "Authentic" if y_pred else "Tampered"

def image_manipulation_detection(image):
    if isinstance(image, str):
        return infer(image)
    elif isinstance(image, np.ndarray):
        Image.fromarray(image).save("temp/user_image.jpg")
        return infer("temp/user_image.jpg")
    else:
        return "Invalid input"

input_image = gr.Image()
output_text = gr.Textbox()

gr.Interface(fn=image_manipulation_detection, inputs=input_image, outputs=output_text, title="Image Manipulation Detection").launch()
