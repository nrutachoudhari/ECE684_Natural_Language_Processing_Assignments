"""
ECE 684 Natural Language Processing
Assignment 5 - Gradient Descent
Nruta Choudhari
"""

"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt
import time

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 1000
    learning_rate = 0.005

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    start_time = time.time()

    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # save the loss for plotting
        losses.append(loss.item())

    # getting the final loss
    final_loss = losses[-1]
    print(f"Final loss: {final_loss}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")
    plot_loss_over_time(losses)

    with torch.no_grad():
        learned_probs = torch.softmax(model.s, dim=0).numpy().flatten()
    optimal_probs = np.array(
        [tokens.count(token) / len(tokens) for token in vocabulary]
    )

    plot_token_probabilities(vocabulary, learned_probs, optimal_probs)


# display results
def plot_token_probabilities(vocabulary, learned_probs, optimal_probs):
    plt.figure(figsize=(10, 5))
    indices = range(len(vocabulary))

    # plotting the learned probabilities
    plt.bar(indices, learned_probs, width=0.4, label="Learned Probs", align="center")

    # plot optimal probabilities
    plt.bar(indices, optimal_probs, width=0.4, label="Optimal Probs", align="edge")

    plt.xticks(indices, vocabulary, rotation=90)
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    plt.title("Comparison of Learned and Optimal Token Probabilities")
    plt.legend()
    plt.show()


def plot_loss_over_time(losses):
    plt.plot(losses, label="Training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss as a function of iteration")
    plt.legend()
    plt.show()


# raise RuntimeError("Remove this error and create visualizations.")  # DO THIS


if __name__ == "__main__":
    gradient_descent_example()
