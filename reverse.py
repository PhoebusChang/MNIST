import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class LabelToImageGenerator(pl.LightningModule):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(10, 10)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()  # outputs in range [-1, 1] like normalized MNIST
        )

    def forward(self, z, labels):
        label_embedding = self.label_embed(labels)
        x = torch.cat([z, label_embedding], dim=1)
        x = self.fc(x)
        return x.view(-1, 1, 28, 28)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        z = torch.randn(x.size(0), self.latent_dim).type_as(x)
        generated = self(z, labels)
        loss = F.mse_loss(generated, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

generator = LabelToImageGenerator()
trainer = pl.Trainer(max_epochs=20, accelerator="auto", devices="auto")
trainer.fit(generator, train_loader)

# === Generate & Save Images ===
def generate_digit_image(label, out_path="generated.png"):
    generator.eval()
    z = torch.randn(1, generator.latent_dim)
    label_tensor = torch.tensor([label])
    with torch.no_grad():
        generated_image = generator(z, label_tensor)
        save_image(generated_image, out_path, normalize=True)
    print(f"Saved digit '{label}' image to {out_path}")

# === Interactive Test ===
while True:
    val = input("Enter digit 0–9 to generate image or type anything else to quit: ")
    if val.isdigit() and 0 <= int(val) <= 9:
        generate_digit_image(int(val))
    else:
        break
