#put new model here
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms as T
import cv2 as cv
from PIL import Image

frame_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225])
])


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(n_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Downsampling
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))

        # Upsampling
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("model_weights.pt", map_location=device)

model = UNet()                # create the model
model.load_state_dict(state_dict)   # load weights
model.to(device)
model.eval()

img = cv.imread("testing/frame0.png")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print("Img RGB shape:", img_rgb.shape)

pil_img = Image.fromarray(img_rgb).convert("RGB")
input_tensor = frame_transform(pil_img).unsqueeze(0).to(device)
print("Input:", input_tensor.shape)

with torch.no_grad():
    pred = model(input_tensor)      # shape: (1, 1, H, W)
    pred_mask = torch.sigmoid(pred)

mask = pred_mask.squeeze().cpu().numpy()
mask_bin = (mask > 0.5).astype("uint8") * 255

# Resize mask to original image resolution (1744 x 934)
mask_bin_resized = cv.resize(mask_bin, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

# Overlay
overlay = img.copy()
overlay[mask_bin_resized == 255] = (0, 255, 0)
cv.imwrite("Original.png", img)
cv.imwrite("Predicted.png", mask_bin)
cv.imwrite("Overlay.png", overlay)
cv.waitKey(0)

model.eval()