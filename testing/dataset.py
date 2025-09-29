import torch 
import torchvision.io as io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
from nnmodel import DynamicModel

class DynamicMotionSet(Dataset):
    def __init__(self, path, transform=None):
        # Make dataset from video folder
        # Path = testing/dataset
        self.frame_dir = path + "/frames"
        self.mask_dir = path + "/masks"
        self.transform = transform
        self.samples = sorted(os.listdir(self.frame_dir))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.samples[idx]

        # Load frame and mask
        frame = Image.open(os.path.join(self.frames_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, fname)).convert("L")

        if self.transform():
            frame = self.transform(frame)
            mask = self.transform(mask)

        return frame, mask

    @property
    def classes(self):
        # pytorch labels are numbers linked to str classes
        return self.data.classes

path = "testing/dataset" # configure pytorch bgsub dataset later
transform = transforms.Compose([
    # resize dataset img size to desired later
    transforms.Resize((480, 480)), #prev 128
    transforms.ToTensor()
])
train_dataset = DynamicMotionSet(path+"/train", transform=transform)
test_dataset = DynamicMotionSet(path+"/test", transform=transform)

# shuffle for training, in-order is fine for test/val
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=False)

for img, mask in train_dataloader:
    print()
#for images, labels in dataloader:
    #break to test if it loaded in properly



# TRAINING LOOP
model = 5 # change to actuall nn
train_loader = "testing/dataset/train" # change to actual train folder
num_epochs = 5
train_losses, val_losses = [], []
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward
        optimizer.step()
        running_loss += loss.item() #* inputs.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

# VALIDATION PHASE
model.eval()
running_loss = 0.0