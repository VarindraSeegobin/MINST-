import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import MinstNet
from DataLoader import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


net = MinstNet().to(device)


cwd = os.getcwd()
loader = DataLoader(cwd + "/data/train-labels.idx1-ubyte",
                    cwd + "/data/train-images.idx3-ubyte",
                    cwd + "/data/t10k-labels.idx1-ubyte",
                    cwd + "/data/t10k-images.idx3-ubyte")

train_images, train_labels = loader.load_training()

train_images = torch.tensor(np.array(train_images), dtype=torch.float32) / 255.0  # Normalize
train_labels = torch.tensor(train_labels, dtype=torch.long)


train_images = train_images.unsqueeze(1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training parameters
batch_size = 64
num_epochs = 30
num_samples = train_images.shape[0]

# Training loop
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    indices = torch.randperm(num_samples)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]

        inputs = train_images[batch_indices].to(device)
        labels = train_labels[batch_indices].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (num_samples // batch_size)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(net.state_dict(), "minstnet_final.pth")
