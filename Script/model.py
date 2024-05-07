import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import glob
import pickle
import time
import psutil
import GPUtil

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64*28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class BrainTumorClassifier:
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        train_data = self.load_folder_data(os.path.join(self.data_dir, 'train'))
        test_data = self.load_folder_data(os.path.join(self.data_dir, 'test'))
        validate_data = self.load_folder_data(os.path.join(self.data_dir, 'validate'))

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        validate_loader = DataLoader(validate_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return train_loader, test_loader, validate_loader

    def load_folder_data(self, folder_path):
        data = []
        label_mapping = {'brain_tumor': 0, 'healthy': 1}
        for label_name in os.listdir(folder_path):
            label = label_mapping[label_name]
            label_folder = os.path.join(folder_path, label_name)
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                image_tensor = torch.load(file_path)
                data.append((image_tensor, label))

        images, labels = zip(*data)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        dataset = TensorDataset(images, labels)
        return dataset

    def train_model(self, train_loader, validate_loader, num_epochs=50, lr=0.001):
        model = CNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        train_accuracies = []  # To store train accuracies
        val_accuracies = []    # To store validation accuracies

        start_time = time.time()  # Record starting time

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            train_loss = train_loss / len(train_loader.dataset)
            train_accuracy = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
            val_loss = val_loss / len(validate_loader.dataset)
            val_accuracy = correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        end_time = time.time()  # Record ending time

        # Calculate total running time
        total_time = end_time - start_time

        # Collect system metrics
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
     

        # Log metrics into a file
        with open('Running_Time.txt', 'w') as f:
            f.write(f'Total Running Time: {total_time} seconds\n')
            f.write(f'CPU Usage: {cpu_usage}%\n')
            f.write(f'Memory Usage: {mem_usage}%\n')

        # Save the model history
        model_history = {'train_losses': train_losses, 'val_losses': val_losses, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
        with open('model_history.pkl', 'wb') as f:
            pickle.dump(model_history, f)

        return model
    
    def evaluate_model(self, model, test_loader):
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return cm, accuracy
