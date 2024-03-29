import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import splitfolders

# Path to the folder containing the dataset
data_frame = pathlib.Path("/Users/estefanos/Desktop/Class Projects/Deep_learing/Group_Project/CSC671_GroupProject/CNN/Brain Tumor Data Set")

try:
    # Split the dataset into training and testing sets
    splitfolders.ratio(data_frame, output='brain', seed=20, ratio=(0.8, 0.2))

    # Define transformations for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])
    # Load the dataset using torchvision datasets.ImageFolder
    train_dataset = datasets.ImageFolder('brain/train', transform=transform)
    test_dataset = datasets.ImageFolder('brain/val', transform=transform)

    # Update the class_to_idx dictionary to map classes to labels
    train_dataset.class_to_idx = {'healthy': 0, 'brain_tumor': 1}
    test_dataset.class_to_idx = {'healthy': 0, 'brain_tumor': 1}

    # Create DataLoader objects to batch and shuffle the data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
except Exception as e:
    print("--------- File created -----------")
    

# Function to display images
def show_image(image_tensor, title):
    image = image_tensor.permute(1, 2, 0)  # Reshape tensor for display
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Plot one image from the training dataset
for images, labels in train_loader:
    show_image(images[0], train_dataset.classes[labels[0]])
    break

# Plot one image from the testing dataset
for images, labels in test_loader:
    show_image(images[0], test_dataset.classes[labels[0]])
    break
