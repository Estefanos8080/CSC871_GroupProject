import os
import sys
# Append the path to your project's root directory
sys.path.append('/Users/estefanos/Desktop/Class Projects/Deep_learing/Group_Project/CSC671_GroupProject/CNN')

# Now you can import the module
from Script.Model import BrainTumorClassifier
import torch
from torch.utils.data import DataLoader


class Train:
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_model(self, num_epochs=50, lr=0.001):
        classifier = BrainTumorClassifier(self.data_dir, self.batch_size, self.num_workers)
        train_loader, test_loader, validate_loader = classifier.load_data()
        model = classifier.train_model(train_loader, validate_loader, num_epochs, lr)
        return model

if __name__ == "__main__":
    data_dir = '/Users/estefanos/Desktop/Class Projects/Deep_learing/Group_Project/CSC671_GroupProject/CNN/Data_Set'
    trainer = Train(data_dir)
    model = trainer.train_model()
