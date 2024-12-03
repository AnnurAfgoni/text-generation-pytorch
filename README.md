# Character-Level RNN Text Generation

## Overview

This project implements a character-level recurrent neural network (RNN) for text generation. The model learns to predict the next character in a sequence of text based on a large corpus of input data. By iteratively predicting characters, the model generates new text sequences resembling the style and structure of the training data.

The primary use cases for such a project include creative writing, generating text in a specific style, or exploring sequence modeling techniques. The project includes data preprocessing, model definition, training, evaluation, and inference modules.

---

### Model

The core of this project revolves around a character-level RNN implemented using PyTorch. The model architecture combines fundamental components designed for sequence prediction tasks:

#### Architecture

1. Input Layer
- The input to the model is a one-hot encoded vector representing the current character.
- The input size equals the total number of unique characters (vocabulary size) in the dataset.

2. LSTM Layers
- The backbone of the model is a 2-layer Long Short-Term Memory (LSTM) network.
- LSTM is chosen for its ability to capture long-term dependencies in sequential data, avoiding the vanishing gradient problem that affects vanilla RNNs.
- Each LSTM layer has 256 hidden units, which allow it to capture complex patterns in character sequences.

3. Dropout Layer

- A dropout layer is applied to the output of the LSTM layers to prevent overfitting. The dropout rate is set to 50%.

4. Fully Connected (FC) Layer

- The LSTM's output is passed through a fully connected layer to project it onto the vocabulary space.
- This layer has a size of (256, vocab_size), where 256 is the size of the LSTM's hidden state, and vocab_size is the number of unique characters in the dataset.
- The output of this layer represents unnormalized log probabilities for each character.

5. Softmax Activation

- During inference, the logits from the FC layer are converted into probabilities using a softmax function. This enables sampling the next character based on the predicted probability distribution.

#### Key Features of the Model

1. Multi-Layer LSTM

- Two LSTM layers allow the model to capture hierarchical patterns in text data. The first layer extracts basic sequential dependencies, while the second refines higher-level patterns.

2. Character-Level Modeling

- The model focuses on individual characters, making it suitable for tasks where token-level granularity is important, such as learning unconventional text styles or processing datasets with non-standard tokens.

3. Dynamic Hidden State Management

- The hidden state is initialized at the beginning of each batch and detached from the computation graph during training to prevent gradients from propagating through earlier iterations.
- This ensures efficient memory usage while retaining temporal context.

4. Generative Capability

- The model can generate text by sampling one character at a time. It takes a "prime" string as input, predicts the next character, appends it to the prime string, and repeats the process iteratively.

5. Flexibility for Training and Inference

- The implementation supports both training (with teacher forcing) and inference (with character sampling). The inference process allows for creative exploration by adjusting parameters like top_k (sampling diversity).

---
### Diagram of Model Architecture

```yaml
Input: One-hot encoded vector of the current character
  |
  V
[LSTM Layer 1]
  |
  V
[Dropout Layer]
  |
  V
[LSTM Layer 2]
  |
  V
[Dropout Layer]
  |
  V
[Fully Connected Layer]
  |
  V
[Softmax Activation] -> Predicted probabilities for each character

```

---
### Training Process

The training process of this character-level RNN differs from traditional models like those used in computer vision. This section explains the unique aspects of training sequence models, particularly character-level RNNs, and why certain design choices, like hidden states, are crucial.

#### How the Network is Trained

1. Input Representation:

- The network processes sequential data character by character.
- Each input sequence is a string of characters of length seq_length (e.g., "The quick br"). The corresponding target is the same sequence, shifted by one character (e.g., "he quick bro").
- Characters are one-hot encoded to form input tensors of shape (batch_size, seq_length, vocab_size).

2. Recurrent Nature of RNNs:

- Unlike regular feedforward models (e.g., CNNs), RNNs require the context from previous time steps. To achieve this, RNNs maintain a hidden state that carries information across time steps within a sequence.
- The hidden state is initialized at the start of each sequence and updated after processing each character.

3. Hidden States:

- The hidden state (`h`) is a tuple of tensors for LSTM layers, consisting of the hidden state (`h_t`) and cell state (`c_t`):
```arduino
h = (h_t, c_t)
```
- The hidden state:
    - Captures information about the sequence seen so far.
    - Enables the network to "remember" patterns across time steps.
- At each training iteration, the hidden state is detached from the computation graph to avoid propagating gradients through the entire sequence history (which could lead to exploding/vanishing gradients).

4. Output and Loss Calculation:

- At every time step, the model predicts the next character in the sequence.
- For a given batch of sequences, the output has the shape `(batch_size * seq_length, vocab_size)` after reshaping, where:
    - `batch_size * seq_length` corresponds to the total number of predictions.
    - `vocab_size` is the probability distribution over characters.
- The target is flattened to match the shape of the predictions.
- **Cross-Entropy Loss** is computed to compare the predicted character probabilities with the true character indices.

5. Teacher Forcing:

- During training, the ground truth character at each time step is fed to the network, even if the previous prediction is incorrect. This is known as teacher forcing.
- It ensures stable training by reducing error propagation, especially in the early stages of training.

6. Backpropagation Through Time (BPTT):

- RNNs use a variation of backpropagation called **Backpropagation Through Time (BPTT)**:
Gradients are computed for each time step in the sequence.
The network's weights are updated based on the cumulative gradients from all time steps.

#### Why Training is Unique

1. Sequential Data Dependency
- Unlike image data (which is processed in parallel), sequences must be processed in order. The output at time t depends on the inputs at time t-1, making the training inherently sequential.

2. Role of the Hidden State

- The hidden state serves as the "memory" of the RNN. Without it, the model would lose context about the sequence, drastically limiting its ability to learn dependencies across time.

3. Variable Sequence Lengths

- In natural language, sequence lengths can vary. Padding is often used for shorter sequences, but this project avoids padding by processing sequences of fixed length (seq_length).
4. Importance of Detaching the Hidden State

- By detaching the hidden state after each batch, the network avoids propagating gradients through the entire sequence history, improving memory efficiency and stability.

5. Generation During Training

- While not explicitly part of training, the generative ability of the model can be evaluated mid-training by "sampling" text. This involves feeding a prime string to the model and allowing it to predict subsequent characters, showcasing how well it has learned the dataset's patterns.
