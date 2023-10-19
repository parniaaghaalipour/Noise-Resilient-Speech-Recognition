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

## Licence
[MIT](https://choosealicense.com/licenses/mit/)

## Feedback

Constructive criticism is absolutely welcome! Feel free to send us your feedback by filing an issue on this repository. Bug reports, feature requests, add-ons or any general improvements are more than welcome.
