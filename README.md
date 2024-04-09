Brain Tumor Classification using CNNs

This project aims to develop a Convolutional Neural Network (CNN) model using PyTorch to accurately classify brain tumors in MRI scans. Early and accurate brain tumor diagnosis is crucial for effective treatment planning and improving patient outcomes. CNNs offer a promising approach for automated tumor detection from medical images.
Approach

key steps:

    Preprocessing:
        
        Data loading using a custom PyTorch data loader.
        Splitting the dataset into training, validation, and testing sets.
        Identifying and removing noise or corrupted images.
        Standardizing image dimensions for consistency.
        
    Building and Training the Model:
        
        Experimenting with different CNN architectures and hyperparameters:
            Number of layers
            Number od epochs
            Optimizer (Adam, SGD)
            Loss function (binary cross-entropy)
        

Results

    Evaluation:
        Model performance will be evaluated on unseen data from the testing set.
        We will use various metrics: Accuracy, Precision, Recall, F1-Score, AUC and LC.

Dataset

We plan to utilize the publicly available Brian Tumor Dataset from Kaggle [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset]. This dataset comprises MRI scans categorized as normal or containing a tumor.


Instructions:

    Clone this repository.
    Install required dependencies (pip install -r requirements.txt).
    Download the brain tumor dataset from Kaggle.
    Place the dataset in the designated directory (modify paths in the code if needed).
    Run the training script (python train.py).
    Evaluate the model's performance using the evaluation script (python evaluate.py).

License
    
    This project is licensed under the MIT License: https://opensource.org/licenses/MIT.

torch  # PyTorch for deep learning
numpy  # Numerical computing library
matplotlib  # Plotting library