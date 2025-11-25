from model import resnet, vgg, alexnet, utils
from dataset import tiny_imagenet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

# ---------- CONFIG ----------
data_dir = "~/scratch/tiny-imagenet-200"
batch_size = 64
epochs = 120
learning_rate = 1e-3
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Config: Done")

# ---------- DATA ------------
dataset = tiny_imagenet.TinyImageNet(data_dir)

train_loader, val_loader = dataset.get_dataloaders(batch_size = 512)

print("Data: Done")

# ---------- MODEL ----------
model_config = utils.ModelConfig(small_dataset = True,num_classes = 200, num_layers = 34)
model = resnet.ResNet(model_config)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Model: Done")

for name, param in criterion.named_parameters():
    print(name, param.device)

summary(model,input_size = (512,3,64,64), device = 'cuda:1')

# ---------- TRAINING LOOP ----------
prof = torch.profiler.profile(
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes = True,
    profile_memory = True,
    with_flops = True,
    with_stack = True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("~/logdir")
)

prof.start()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        prof.step()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # --------- VALIDATION ---------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct/total:.2f}%")
    
prof.stop()

print("Training complete.")
