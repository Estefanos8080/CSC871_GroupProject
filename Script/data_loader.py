import os
import random
import shutil
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torch

class BrainTumorDataset:
    def __init__(self, data_dir, train_ratio=0.7, test_ratio=0.15, validate_ratio=0.15, batch_size=32):
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validate_ratio = validate_ratio
        self.batch_size = batch_size

    def preprocess_data(self):
        # Define transformations for preprocessing images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a consistent size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])

        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        total_len = len(dataset)
        
        train_size = int(self.train_ratio * total_len)
        test_size = int(self.test_ratio * total_len)
        validate_size = total_len - train_size - test_size
        
        # Shuffle the indices
        indices = list(range(total_len))
        random.shuffle(indices)

        # Split indices into train, test, and validate sets
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]
        validate_indices = indices[train_size + test_size:]

        # Define labels
        label_names = {0: 'brain_tumor', 1: 'healthy'}

        # Move images to train, test, and validate folders
        for idx, split_indices in enumerate([(train_indices, 'train'), (test_indices, 'test'), (validate_indices, 'validate')]):
            split_indices, split_name = split_indices
            for i in split_indices:
                label = dataset.targets[i]
                label_name = label_names[label]
                image, _ = dataset[i]  # Apply transformation here
                new_path = os.path.join(self.data_dir, split_name, label_name, f'{i}.pt')
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                torch.save(image, new_path)  # Save the transformed image as a PyTorch tensor

    def plot_images(self, labels):
        # Define labels
        label_names = {0: 'brain_tumor', 1: 'healthy'}

        # Loop through train, test, and validate folders
        for split_name in ['train', 'test', 'validate']:
            fig, axes = plt.subplots(1, len(labels), figsize=(15, 5))
            for i, label in enumerate(labels):
                img_folder = os.path.join(self.data_dir, split_name, label_names[label])
                img_file = random.choice(os.listdir(img_folder))  # Select a random image
                img_path = os.path.join(img_folder, img_file)
                image = torch.load(img_path)  # Load the PyTorch tensor

                # Plot the image with label
                axes[i].imshow(transforms.ToPILImage()(image))
                axes[i].set_title(f'{label_names[label]}')
                axes[i].axis('off')
            plt.suptitle(f'{split_name.capitalize()} Images')
            plt.show()


data_dir = '/Users/estefanos/Desktop/Class Projects/Deep_learing/Group_Project/CSC671_GroupProject/CNN/Data_Set'
labels = [0, 1]  # Labels for healthy and brain_tumor classes
dataset = BrainTumorDataset(data_dir)
dataset.preprocess_data()

############ TO PLOT THE MRI SCANS UNCOMMENT THE LINE BELOW #############
# dataset.plot_images(labels)
