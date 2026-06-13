# Gradient Descent

## Overview

This assignment extends a provided PyTorch unigram language model to add training visualizations and explores the effect of learning rate and iteration count on convergence. The model is trained on character-level data from Jane Austen's *Sense and Sensibility* and evaluated by comparing learned probabilities against optimal MLE estimates.

## Files

- `unigram_pytorch.py` — extended implementation with loss tracking and visualizations
- `unigram_pytorch_original.py` — provided starter file
- `Gradient_Descent_Nruta_Choudhari.pdf` — written responses covering model explanation, hyperparameter analysis, and document classification extension

## What the Model Does

The network is a unigram character language model. It takes one-hot encoded characters as input and learns a probability distribution over a 28-token vocabulary (a-z, space, UNK) by minimizing negative log likelihood via gradient descent. The learned parameter `s` is passed through LogSoftmax to produce a valid log probability distribution, which is then scored against the observed token counts.

The optimal probabilities are simply the empirical character frequencies in the training text, which the model converges toward through training.

## Hyperparameter Experiments

| Learning Rate | Iterations | Training Time | Final Loss |
|---|---|---|---|
| 0.001 | 500 | 1.01s | 2,081,818 |
| 0.01 | 1000 | 1.9s | 1,957,039 |
| 0.1 | 5000 | 9.88s | 1,956,527 |

Learning rate 0.01 with 1000 iterations is the best trade-off: the loss nearly matches the 5000-iteration run at a fraction of the training time. Increasing to lr=0.1 yields diminishing returns and 5x the compute.

## Visualizations

Two plots are generated per run:

**Loss over iterations:** Shows the training loss curve from initialization to convergence. At lr=0.01 the loss drops sharply in the first ~200 iterations then flattens, indicating fast early learning.

**Learned vs. optimal token probabilities:** Side-by-side bar chart comparing the model's learned distribution to the empirical character frequencies. At 1000 iterations the learned bars closely track the optimal bars for high-frequency characters like `e`, `t`, `a`, and space.

## Extending to Document Classification

To adapt this model for document classification, the vocabulary would shift from characters to words, and each document would be represented as a word count vector rather than a sequence of one-hot character encodings. Separate unigram models would be trained per class label (e.g., sports, politics), and at inference time a new document would be assigned to the class whose model assigns it the highest probability. Using n-gram tokens instead of unigrams would further improve classification by capturing phrase-level context.

## Libraries Used

`torch`, `numpy`, `matplotlib`, `nltk`