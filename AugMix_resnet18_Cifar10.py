
# Execute this code block to install dependencies when running on colab
try:
    import torch
except:
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

    !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import AugMix
import numpy as np

# Define the AugMix transformation
def random_augmix_transform():
    severity = np.random.randint(1, 4)  # Random severity level (1, 2, or 3)
    width = np.random.randint(1, 4)  # Random mixture width (1, 2, or 3)
    depth = np.random.randint(1, 4)  # Random chain depth (1, 2, or 3)

    augmix_transform = transforms.Compose([
        AugMix(severity=severity, mixture_width=width, chain_depth=depth, alpha=1.0, all_ops =True),
        transforms.ToTensor(),
    ])

    return augmix_transform

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True)

# Create a list to store augmented images and labels
augmented_data = []

# Apply random AugMix transformation to each image and store augmented images and labels
for image, label in train_dataset:
    augmix_transform = random_augmix_transform()
    augmented_image = augmix_transform(image)
    augmented_data.append((augmented_image, label))

transform_test = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform = transform_test)

batch_size =256
#
train_loader = DataLoader(augmented_data, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch 
import torch.nn.functional as F
from torch import nn
import torchbearer
from torchbearer import Trial
from torch import optim
from torchvision.models import resnet18

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm  # Import tqdm for the progress bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False, num_classes=10).to(device)

# Define the optimizer and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    # Use tqdm for the progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update the tqdm description with the current loss
            t.set_postfix(loss=running_loss/len(t))
    
    # Update the learning rate
    scheduler.step()

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loss = criterion(outputs, labels)
        test_loss += loss.item()

accuracy = 100 * correct / total
average_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
