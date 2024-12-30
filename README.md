# Brain Tumor Detection Using Deep Learning

## Overview
This project aims to classify brain MRI images into multiple categories, including tumor and non-tumor classes, using a convolutional neural network (CNN). The model was trained and fine-tuned using a dataset from Kaggle, leveraging data augmentation and callbacks to achieve high performance.

---

## Features
- **Data Augmentation**: Rescaling and preprocessing techniques for robust training.
- **Model Architecture**: A CNN with multiple layers of convolution, pooling, dropout, and batch normalization to extract features and minimize overfitting.
- **Fine-Tuning**: Learning rate scheduling and early stopping to improve training efficiency and avoid overfitting.
- **Visualization**: Display of dataset samples and performance metrics (accuracy and loss) over epochs.

---

## Dataset
- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Structure**: The dataset is divided into training and testing directories with images organized by class.

---

## Requirements
Install the necessary libraries before running the code:

```bash
pip install tensorflow matplotlib numpy
```

Other dependencies include:
- Google Colab (for executing the script)
- Keras (for deep learning)
- Scikit-learn (for computing class weights)

---

## Model Architecture
The model is a sequential CNN with the following layers:
1. **Convolutional Layers**: Extract spatial features using 32, 64, 128, and 256 filters.
2. **Pooling Layers**: Reduce spatial dimensions with max pooling.
3. **Batch Normalization**: Stabilize and speed up training.
4. **Dropout**: Prevent overfitting with dropout rates of 0.3–0.6.
5. **Dense Layers**: Fully connected layers for classification.
6. **Output Layer**: Softmax activation for multi-class predictions.

---

## Training Strategy
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Categorical crossentropy for multi-class classification.
- **Metrics**: Accuracy.
- **Callbacks**:
  - **ReduceLROnPlateau**: Reduces learning rate if validation loss plateaus.
  - **EarlyStopping**: Stops training if no improvement is seen in validation loss for 8 epochs.

---

## Usage
1. **Download Dataset**: Download and unzip the dataset.
2. **Run Training Script**: Train the model using the provided code.
3. **Evaluate Model**: Evaluate the trained model on unseen data to check its performance.

---

## Performance
- **Accuracy**: Achieved high accuracy (99.1% on training data).
- **Loss**: Low training and validation loss.
- **Visualization**: Performance trends are visualized using accuracy and loss plots.

---

## Results
The model demonstrated high classification accuracy with robust training and validation performance. The visualizations indicate consistent improvement in accuracy and reduction in loss over epochs.

---

## Future Enhancements
- Add more advanced data augmentation techniques.
- Experiment with transfer learning for improved feature extraction.
- Optimize hyperparameters for further performance gains.

---

## Acknowledgments
- **Dataset**: Thanks to Kaggle for providing the dataset.
- **Tools**: TensorFlow, Keras, and Colab for enabling the implementation.

---
