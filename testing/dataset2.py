import torch 
import torchvision.io as io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
import numpy as np
from nnmodel import DynamicModel
import torchvision.transforms as T
import torch.nn as nn

frame_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet
                std=[0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),  # gives float [0,1]
])


class DynamicMotionSet(Dataset):
    def __init__(self, path, transform=None, mask_transform=None):
        self.frame_dir = os.path.join(path, "frames")
        self.mask_dir = os.path.join(path, "masks")
        self.transform = transform
        self.mask_transform = mask_transform

        # Get all frame names
        self.samples = sorted(os.listdir(self.frame_dir))

        # Keep only frames that have a corresponding mask
        valid_samples = []
        for f in self.samples:
            mask_name = f.replace("frame", "mask")
            if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                valid_samples.append(f)

        self.samples = valid_samples
        if len(self.samples) == 0:
            raise RuntimeError(f"No matching frame/mask pairs found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]

        # --- Load frame ---
        frame = Image.open(os.path.join(self.frame_dir, fname)).convert("RGB")

        # --- Load matching mask ---
        mask_name = fname.replace("frame", "mask")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")  # still PIL

        # --- Apply transforms ---
        if self.transform is not None:
            frame = self.transform(frame)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)  # expects PIL image
        else:
            mask = T.ToTensor()(mask)

        # --- Convert to binary (0/1) ---
        mask = (mask > 0.5).float().squeeze(0)  # [H, W] float

        return frame, mask

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

dataset = DynamicMotionSet("testing/dataset/train", transform=frame_transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = UNet(n_channels=3, n_classes=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for frames, masks in dataloader:
        frames = frames.to(device)
        masks = masks.to(device).float()  # BCE expects float (0.0 or 1.0)

        optimizer.zero_grad()
        outputs = model(frames)  # [B, 1, H, W]
        outputs = outputs.squeeze(1)  # [B, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

model.eval()
with torch.no_grad():
    frame, _ = dataset[0]
    frame = frame.unsqueeze(0).to(device)
    pred = torch.sigmoid(model(frame))[0,0]  # sigmoid for binary
    pred_mask = (pred > 0.5).cpu().numpy()  # binary mask


torch.save(model.state_dict(), 'model_weights.pt')
