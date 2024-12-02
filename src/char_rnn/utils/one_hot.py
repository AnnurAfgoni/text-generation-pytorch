import numpy as np


def one_hot_encode(sequence, n_labels):
    """
    Apply one-hot encoding to a sequence.

    Args:
        sequence (np.ndarray): Input sequence of integers.
        n_labels (int): Number of unique characters.

    Returns:
        np.ndarray: One-hot encoded sequence.
    """
    one_hot = np.zeros((len(sequence), n_labels), dtype=np.float32)
    one_hot[np.arange(len(sequence)), sequence] = 1.0
    return one_hot
