import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import splitfolders

class BrainDataset:
    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def preprocess_data(self):
        try:
            # Check if the brain folder exists, if not, create it
            if not (self.data_dir / "brain").exists():
                splitfolders.ratio(self.data_dir, output='brain', seed=20, ratio=(0.7, 0.1, 0.2))

            # Resize images to a consistent size and Convert images to PyTorch tensors
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  
                transforms.ToTensor(), 
            ])

            # Load the dataset using torchvision datasets.ImageFolder
            self.train_dataset = datasets.ImageFolder(self.data_dir / '/brain/train', transform=transform)
            self.val_dataset = datasets.ImageFolder(self.data_dir / 'brain/val', transform=transform)
            self.test_dataset = datasets.ImageFolder(self.data_dir / 'brain/test', transform=transform)

            # Update the class_to_idx dictionary to map classes to labels
            self.train_dataset.class_to_idx = {'healthy': 0, 'brain_tumor': 1}
            self.val_dataset.class_to_idx = {'healthy': 0, 'brain_tumor': 1}
            self.test_dataset.class_to_idx = {'healthy': 0, 'brain_tumor': 1}

            # Create DataLoader objects to batch and shuffle the data
            self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        except Exception as e:
            print("Error occurred:", e)

    def get_dimensions(self):
        if self.train_loader is not None and self.val_loader is not None:
            for key, value in {'Training data': self.train_loader, "Val data": self.val_loader}.items():
                for X, y in value:
                    print(f"{key}:")
                    print(f"Shape of X : {X.shape}")
                    print(f"Shape of y: {y.shape} {y.dtype}\n")
                    break

    def show_image(self, image_tensor, title):
        image = image_tensor.permute(1, 2, 0)  # Reshape tensor for display
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def plot_samples(self):
        if self.train_loader is not None and self.val_loader is not None:
            # Plot one image from the training dataset
            for images, labels in self.train_loader:
                self.show_image(images[1], self.train_dataset.classes[labels[1]])
                break

            # Plot one image from the testing dataset
            for images, labels in self.val_loader:
                self.show_image(images[0], self.test_dataset.classes[labels[0]])
                break


data_dir = "/home/ubuntu/Project/GP/CSC671_GroupProject/CNN/Data_Set"
brain_dataset = BrainDataset(data_dir)
brain_dataset.preprocess_data()
brain_dataset.get_dimensions()
# brain_dataset.plot_samples()
