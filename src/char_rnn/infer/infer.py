import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from char_rnn.utils import one_hot_encode
from char_rnn.model import CharRNN


class InferenceModule:
    def __init__(
        self,  
        tokens,
        char2int, 
        int2char,
        model_path: str=None, 
        device: str="cpu"
    ):
        """
        Initializes the inference module.

        Args:
            model (nn.Module): The trained character-level RNN model.
            char2int (dict): Mapping from characters to integers.
            int2char (dict): Mapping from integers to characters.
            device (str): Device to run inference on ("cuda" or "cpu").
        """
        super().__init__()

        path_to_save = str(self._get_cache_dir()) + "/model.ckpt"
        if model_path is None:
            downloader(path_to_save)
            self.model_path = path_to_save
        else:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError("Model does not exist, check your model location and read README for more information")

        self.tokens = tokens
        self.char2int = char2int
        self.int2char = int2char

        self.device = device
        self.model = self._get_model()

    def _get_model(self):
        """
        Returns the loaded model.
        """
        model = CharRNN(self.tokens)
        _state_dict = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )
        state_dict = {k.replace("model.", ""):v for k,v in _state_dict["state_dict"].items()}
        model.load_state_dict(state_dict)
        return model
    
    def _get_cache_dir(self):
        if sys.platform.startswith("win"):
            cache_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "OnepieceClassifyCache"
        else:
            cache_dir = Path.home() / ".cache" / "OnepieceClassifyCache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def preprocess(self, char):
        """
        Prepares the input character for the model.

        Args:
            char (str): A single character to process.

        Returns:
            torch.Tensor: One-hot encoded input tensor.
        """
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.char2int))  # Assuming one_hot_encode is defined
        return torch.from_numpy(x).to(self.device)

    def forward(self, char, hidden, top_k=None):
        """
        Makes a single character prediction.

        Args:
            char (str): Input character.
            hidden (tuple): Hidden state of the model.
            top_k (int, optional): Number of top characters to sample from. Defaults to None.

        Returns:
            tuple: Predicted next character and updated hidden state.
        """
        inputs = self.preprocess(char).unsqueeze(0)
        # hidden = tuple([each.data for each in hidden])
        hidden = tuple([h[:, :inputs.size(0), :].contiguous() for h in hidden])

        # Forward pass
        out, hidden = self.model(inputs, hidden)

        # Get probabilities
        p = F.softmax(out, dim=1).data.cpu().numpy().squeeze()
        # p = p.cpu().numpy().squeeze()

        # Get top characters
        if top_k is None:
            top_ch = np.arange(len(self.char2int))
        else:
            p, top_ch = torch.topk(torch.tensor(p), top_k)
            p = p.numpy()
            top_ch = top_ch.numpy()

        p = p / p.sum()

        # Select next character
        char = np.random.choice(top_ch, p=p / p.sum())
        return self.int2char[char], hidden

    def postprocess(self, generated_text):
        """
        Handles any post-processing on generated text.

        Args:
            generated_text (str): The raw generated text.

        Returns:
            str: Cleaned or formatted text.
        """
        # No cleaning for now; directly return the text
        return generated_text

    def generate(self, size, prime="The", top_k=None):
        """
        Generates a sequence of characters.

        Args:
            size (int): Number of characters to generate.
            prime (str): Prime sequence to start generation with.
            top_k (int, optional): Number of top characters to sample from. Defaults to None.

        Returns:
            str: Generated text sequence.
        """
        self.model.eval()
        hidden = self.model.init_hidden(1)
        chars = [ch for ch in prime]

        # Prime the model
        for char in prime:
            _, hidden = self.forward(char, hidden, top_k=top_k)

        # Generate new characters
        for _ in range(size):
            char, hidden = self.forward(chars[-1], hidden, top_k=top_k)
            chars.append(char)

        return self.postprocess("".join(chars))
