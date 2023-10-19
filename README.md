# Noise Resilient Speech Recognition


![F08CB238-92EE-4EFF-8867-A7AC8250CCA3](https://github.com/parniaaghaalipour/Noise-Resilient-Speech-Recognition/assets/141918224/6558748b-292f-46cd-b09b-79e84be1c4b7)


## Introduction

This repository presents a Python implementation for Speech Recognition using PyTorch. Specifically, the code showcases the usage of Transformers, an advanced Neural Network architecture that is increasingly being adopted for tasks involving sequential data processing, such as Language Translation, Text Summarization and in our case Speech Recognition.

Transformers employ the concept of 'Attention' to understand and remember relationships between different positions in the input sequence, improving performance significantly compared to traditional models. 

In layman's terms, Transformers are like more attentive students; they pay attention to the specifics (individual words) and the context (the overall sentence structure) simultaneously, resulting in a highly accurate understanding.


## Requirements
- Python 3.5 or above for scripting.
- PyTorch 1.3.0 or above as it is the backbone library for all computations in this model.

## Usage

Speaking in terms of a black box, you need to simply provide the model with preprocessed speech features that you want it to learn from. Before that, you'll need to set some parameters such as the dimensionality of the input, number of attention layers, level of dropout, etc. Here's how:

First, define your hyperparameters: 
- `input_dim`: The dimensionality of your input. This stands for the different 'features' or distinct pieces of information your input data carries.
- `num_heads`: Known as 'attention heads', these are separate learning modules of the Transformer model, each learning different 'aspects' of the input data.
- `num_layers`: The number of layers in the Transformer model. Multiple layers allow the model to learn complex, multi-level abstract patterns in the data.
- `dropout`: A regularization technique to prevent overfitting in the model. It defines the ratio of nodes to randomly exclude from each update cycle.
- `output_dim`: The dimensionality of output. This is the number of classes/categories your model will predict between.

Then, declare an instance of the Transformer model like this:

```python
model = TransformerModel(input_dim, num_heads, num_layers, dropout, output_dim).to(device)
```

Feed your speech features (preprocessed appropriately) to the model in this manner:

```python
output = model(src)
```

## Model Description

Let's break down the model you'll be working with:

- **Embedding Layer**: This layer is responsible for converting your input data into a format (lower dimensional vectors) that the neural network can effectively process.
- **Positional Encoding Layer**: This module adds information about sequence order to the input data, ensuring that the model retains the context of the sequence.
- **Transformer Encoder Layer**: This is the core of the model, consisting of multiple layers of attention mechanisms (to pay attention to different aspects of the input) and feed-forward networks (for the actual learning).
- **Final Decoder Layer**: Lastly, the model uses a final decoder layer to map the output of the Transformer encoder to the required output dimensions.

The most critical feature of the Transformer architecture is the implementation of 'Attention' mechanism. Unlike conventional models that process the input sequence part by part or ignore the sequence altogether, Transformer model processes the entire sequence at once and in parallel. This means it handles all positions simultaneously and more attentively.

## Training
This code contains the training script for a speech recognition system. The input speech data is processed by using Connect-Times (CTC) loss which is commonly used in speech recognition tasks. After each training epoch, the loss is written to tensorboard for tracking.

## Usage

You can determine the number of epochs, learning rate, momentum, weight decay and batch size by passing command-line arguments when running the script:

```bash
python train.py --epochs 15 --learning_rate 0.001 --momentum 0.8 --weight_decay 0.0006 --batch_size 64
```

If no command-line argument is passed, the script uses the default values:

- Epochs: 10
- Learning rate: 0.01
- Momentum: 0.9
- Weight decay: 0.0005
- Batch size: 32

## Data

This script assumes data is loaded using a custom DataLoader. The loader is responsible for taking in a batch size argument and a path to train and test datasets.

Please replace 'train_path' and 'test_path' with actual paths to your train and test data.

## Model

The Transformer Model is created with arguments: input_dim, num_heads, num_layers, dropout, output_dim. Please replace these values with your actual values.

The model is sent to GPU if one is available, and trained using SGD optimizer with the given learning rate, momentum and weight decay. The learning rate is scheduled to decrease every 7 epochs by a factor of 0.1.

## Decoding

Decoding is performed by a custom GreedyDecoder. Replace this with decoding method suitable for your task, if needed.

## Saving the model

The script saves the trained model weights after the training process to 'model.pth'. 

## TensorBoard

Tensorboard is used to track the training loss. If you have Tensorboard installed, you can start it by running:

```bash
tensorboard --logdir=runs


## Licence
[MIT](https://choosealicense.com/licenses/mit/)

## Feedback

Constructive criticism is absolutely welcome! Feel free to send us your feedback by filing an issue on this repository. Bug reports, feature requests, add-ons or any general improvements are more than welcome.
