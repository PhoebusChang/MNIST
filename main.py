from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from PIL import Image


# transformation pipeline for preprocessing the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load training & testing MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class SimpleNN(pl.LightningModule):
    def __init__(self):
        # Calls the constructor of the parent class
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        # Forward pass
        out = self(x)
        # Compute loss
        loss = self.loss_fn(out, y)
        # Log the loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)
    
    def configure_optimizers(self):
        # Using Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
model = SimpleNN()
trainer = pl.Trainer(max_epochs=20, accelerator="auto", devices="auto")
trainer.fit(model, train_loader, test_loader)

def predict_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# === Interactive Test ===
while input("Press [Enter] to predict 'test.png' or type anything to quit: ") == "":
    print(predict_image("test.png"))
# === Save and Load Model ===
trainer.save_checkpoint("mnist_model.ckpt")