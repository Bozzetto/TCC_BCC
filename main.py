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
epochs = 2
learning_rate = 1e-3
device = torch.device('cuda:1')
torch.cuda.set_device('cuda:1')
print("Config: Done")

# ---------- DATA ------------
dataset = tiny_imagenet.TinyImageNet(data_dir)

train_loader, val_loader = dataset.get_dataloaders(batch_size = 128)

print("Data: Done")

# ---------- MODEL ----------
model_config = utils.ModelConfig(small_dataset = True,num_classes = 200, num_layers = 34)
model = resnet.ResNet(model_config)
model = model.to(device)

criterion = nn.CrossEntropyLoss().cuda(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Model: Done")

summary(model,input_size = (512,3,64,64), device = 'cuda:1')

# ---------- TRAINING LOOP ----------
prof = torch.profiler.profile(
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    schedule = torch.profiler.schedule(
        wait = 1,
        warmup = 4,
        active = 10,
        repeat = 1 
    ),
    record_shapes = True,
    with_modules = True,
    profile_memory = True,
    with_flops = True,
    with_stack = True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir")
)

prof.start()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking = True), labels.to(device, non_blocking = True)

        optimizer.zero_grad(set_to_none = True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prof.step()

    # --------- VALIDATION ---------
    #model.eval()
    #correct = 0
    #total = 0

    #with torch.no_grad():
    #    for images, labels in val_loader:
    #        images, labels = images.to(device), labels.to(device)
    #        outputs = model(images)
    #        _, predicted = outputs.max(1)
    #        total += labels.size(0)
    #        correct += (predicted == labels).sum()
    #
    #accuracy = (correct.float() / total) * 100
    #print(f"Validation Accuracy: {accuracy.item():.2f}%")
    
prof.stop()

print("Training complete.")
