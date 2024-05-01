Brain Tumor Classification using CNNs

This project aims to develop a Convolutional Neural Network (CNN) model using PyTorch to accurately classify brain tumors in MRI scans. 

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
        Model performance is evaluated on unseen data from the testing set.
        We will use various metrics: Accuracy, Precision, Recall, F1-Score, AUC and LC.

Dataset

We use the publicly available Brian Tumor Dataset from Kaggle [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset]. This dataset comprises MRI scans categorized as Normal or Brain Tumor.


Instructions:

    Clone this repository.
    Install required dependencies (pip install -r requirements.txt).
    Download the brain tumor dataset from Kaggle.
    Place the dataset in the designated directory.
    Run the training script (python train.py).
    

License
    
    This project is licensed under the MIT License: https://opensource.org/licenses/MIT.
