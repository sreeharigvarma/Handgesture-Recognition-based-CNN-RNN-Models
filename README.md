
# Hand Gesture Recognition using CNN and RNN

This project aims to recognize hand gestures using a deep learning model that combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The model is trained to classify different hand gestures from a dataset of images.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Understanding the Dataset](#understanding-the-dataset)
- [Objective](#objective)
- [Consolidated Final Models](#consolidated-final-models)
- [Observations](#observations)
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

## Problem Statement
As a data scientist at a home electronics company which manufactures state-of-the-art smart televisions, we want to develop a cool feature in the smart-TV that can recognize five different gestures performed by the user, which will help users control the TV without using a remote.
- **Thumbs up:** Increase the volume.
- **Thumbs down:** Decrease the volume.
- **Left swipe:** 'Jump' backwards 10 seconds.
- **Right swipe:** 'Jump' forward 10 seconds.
- **Stop:** Pause the movie.

## Understanding the Dataset
The training data has hundreds of videos, each one 2-3 seconds long and split into 30 frames. Different people recorded these videos with a webcam, doing one of the five gestures - like the smart TV will use. These videos belong to one of the five classes.

## Objective
We need to train various models on the 'train' folder that can identify the action in each sequence or video and do well on the 'val' folder too. The final test folder is hidden - we will test the final model's performance on the 'test' set.

## Consolidated Final Models
| Model Name | Model Type | No. of Params | Augment Data | Model Size (MB) | Highest Validation Accuracy | Corresponding Training Accuracy | Remarks |
|------------|------------|---------------|--------------|-----------------|-----------------------------|---------------------------------|---------|
| Conv3D Model | Conv3D | 1,736,389 | Yes | 20 | 0.31 | 0.69 | Test model. |
| Model–1 | Conv3D | 11,170,611 | Yes | 41.8 | 0.29 | 0.97 | Highly overfitting |
| Model–2 | Conv3D | 3,638,981 | Yes | 20.3 | 0.43 | 0.74 | Improved in terms of overfitting |
| Model–3 | Conv3D | 1,762,613 | Yes | 19.1 | 0.2 | 0.63 | Worst model. Very high overfitting |
| Model–4 | CNN-LSTM | 1,657,445 | Yes | 32.4 | 0.47 | 0.89 | Still problem with validation score |
| Model–5 | CNN-LSTM with GRU | 2,573,925 | Yes | 29.6 | 0.25 | 0.75 | Not good, overfitting |

## Observations
- Overfitting was a major issue, which we could not address completely.
- CNN+LSTM based model had better performance than Conv3D.
- Data augmentation helped in overcoming the problem of overfitting which our initial version of model was facing.
- It was observed that as the number of trainable parameters increase, the model takes much more time for training.
- Increasing the batch size greatly reduces the training time but this also has a negative impact on the model accuracy. If we want our model to be ready in a shorter time span, choose a larger batch size, else you should choose a lower batch size if you want your model to be more accurate.

### Suggestions for Improvement
- CNN-LSTM appears to be a good choice. Trainable parameters of a GRU are far less than that of an LSTM, therefore it would have resulted in faster computations. However, its effect on the validation accuracies could be checked to determine if it is actually a good alternative over LSTM.
- Experimenting with other combinations of hyperparameters like activation functions (ReLU, Leaky ReLU, mish, tanh, sigmoid), other optimizers like Adagrad() and Adadelta() can further help develop better and more accurate models. Experimenting with other combinations of hyperparameters like the filter size, paddings, stride length, batch normalization, dropouts etc. can further help improve performance.

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

## Contributors
- Sreehari G Varma
- Bhargav [GitHub](https://github.com/bhagmuniverse)
