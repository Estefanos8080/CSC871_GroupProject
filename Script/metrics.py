
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, roc_auc_score
import numpy as np
import seaborn as sns

model.eval()  # Set the model to evaluation mode

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Containers for true labels and predictions
true_labels = []
predicted_labels = []

# Iterate over the test dataset
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    # Convert output probabilities to predicted class (0 or 1)
    _, predicted = torch.max(outputs, 1)

    # Store predictions and true labels
    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(predicted.cpu().numpy())

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)

# Compute the F1 score
f1 = f1_score(true_labels, predicted_labels)
print("F1 Score:", f1)

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("AUC-ROC:", roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes, rotation=45)
    plt.show()


plot_confusion_matrix(conf_matrix, classes=[0, 1])

with open('model_history.pkl', 'rb') as f:
    model_history = pickle.load(f)

# Extracting data from the dictionary
train_losses = model_history['train_losses']
val_losses = model_history['val_losses']
train_accuracies = model_history['train_accuracies']
val_accuracies = model_history['val_accuracies']

plt.figure()
plt.plot([0, 1, 2, 3, 4], train_losses, label='Train Loss')
plt.plot([0, 1, 2, 3, 4], val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('Train vs. Validation Loss')
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

plt.figure()
plt.plot([0, 1, 2, 3, 4], train_accuracies, label='Train Accuracy')
plt.plot([0, 1, 2, 3, 4], val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs. Validation Accuracy')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
