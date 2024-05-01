import pickle
import matplotlib.pyplot as plt

# Load model history from pickle file
with open('/Users/estefanos/Desktop/Class Projects/Deep_learing/Group_Project/CSC671_GroupProject/Script/model_history.pkl', 'rb') as f:
    model_history = pickle.load(f)

# Extract training and validation losses
train_loss = model_history['train_accuracies']
val_loss = model_history['val_accuracies']

# Plot training and validation loss
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation accuracies')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
