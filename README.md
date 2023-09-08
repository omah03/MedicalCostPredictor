
```markdown
# Feedforward Neural Network for Regression

This repository contains the implementation and training of a feedforward neural network for a regression task using TensorFlow and Keras.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Overview

In this project, we build and train a feedforward neural network to perform a regression task. The model is designed to predict a continuous target variable based on a set of input features.

## Dependencies

- TensorFlow
- Python 3.x
- Pandas (for data preprocessing)
- Matplotlib (for visualization, if needed)

You can install the required packages using pip:

```bash
pip install tensorflow pandas matplotlib
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Prepare your dataset and preprocess it as needed. Make sure you have a dataset with features (X) and a target variable (y).

3. Modify the code to load your dataset and adapt the model architecture as necessary.


## Model Architecture

The feedforward neural network consists of the following layers:

- Input Layer
- Hidden Layer 1: Dense layer with 100 neurons and ReLU activation
- Hidden Layer 2: Dense layer with 10 neurons and ReLU activation
- Output Layer: Dense layer with 1 neuron (typical for regression tasks)

The model is compiled with Mean Absolute Error (MAE) as the loss function and the Adam optimizer.

```python
import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])

# Compile the model
model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["mae"]
)
```

## Training

The model is trained with early stopping to prevent overfitting. The learning rate is scheduled to adjust during training for better convergence.

## Results

- Training history (loss curve)

 ![Screenshot 2023-09-08 at 13 22 10](https://github.com/omah03/MedicalCostPredictor/assets/96381116/6b09a64c-fef4-4bec-93b7-f3a06563f0c5)

- Evaluation metrics
![Screenshot 2023-09-08 at 13 22 30](https://github.com/omah03/MedicalCostPredictor/assets/96381116/6b70685e-3606-4aeb-b15a-e7da4c05ac0f)

- Model predictions
![Screenshot 2023-09-08 at 13 22 43](https://github.com/omah03/MedicalCostPredictor/assets/96381116/8776146b-6876-492a-94d8-888153036bb7)

