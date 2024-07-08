
# Hand Gesture Recognition using CNN and RNN

This project aims to recognize hand gestures using a deep learning model that combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The model is trained to classify different hand gestures from a dataset of images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Hand gesture recognition is a popular area in computer vision with applications in various fields such as human-computer interaction, sign language recognition, and virtual reality. This project implements a hand gesture recognition system using a combination of CNN and RNN architectures to leverage the spatial and temporal features of the gestures.

## Dataset
The dataset used for training and testing the model consists of hand gesture images. Each image is labeled with the corresponding gesture class. The dataset is split into training, validation, and test sets.

## Model Architecture
The model architecture combines CNN and RNN layers:
- **CNN Layers:** Extract spatial features from the images.
- **RNN Layers:** Capture the temporal dependencies between the frames of the gesture.

## Training
The model is trained using the following steps:
1. Preprocessing the images (resizing, normalization).
2. Defining the CNN-RNN architecture.
3. Compiling the model with appropriate loss function and optimizer.
4. Training the model on the training set and validating on the validation set.

## Evaluation
The trained model is evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score.

## Results
The model's performance is visualized using plots of training and validation accuracy/loss over epochs. The confusion matrix is used to analyze the model's classification performance.

## Installation
To run this project, you need to have Python and the required libraries installed. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
To train and evaluate the model, run the following command:

```bash
jupyter notebook Handgesture_CNN_RNN.ipynb
```

This will open the Jupyter Notebook where you can run the cells step by step.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.
