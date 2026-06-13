# Long Short-Term Memory Networks (LSTMs)

## Overview

This is a purely theoretical assignment consisting of two proofs/derivations about LSTM architecture and manually designed weights. No code was required.

## File

- `Nruta_LSTM_Assignment.pdf` — handwritten solutions to both problems

---

## Problem 1: Simple RNN as a Subset of LSTM

The goal is to show that a simple RNN with sigmoid nonlinearity is a special case of an LSTM by identifying specific weight and bias settings that make the LSTM behave identically to the RNN.

The key insight is that the LSTM has three gates (forget, input, output) that control information flow. To recover the simple RNN, all three gates are forced open by setting their biases to infinity and all related weights to zero, so each gate always outputs 1. The candidate cell state uses sigmoid instead of tanh to match the RNN's activation function. With these settings:

- Forget gate ft = 1: the full previous cell state is carried forward with no forgetting
- Input gate it = 1: the new candidate state is fully incorporated
- Output gate ot = 1: the hidden state directly reflects the full cell state
- Cell state update simplifies to ct = ct-1 + candidate, matching the RNN's recurrent behavior
- Hidden state ht = sigmoid(Wig * xt + big + Whg * ht-1 + bhg), which exactly matches the simple RNN update equation

## Problem 2: LSTM Weights for Sentiment Scoring

The task is to manually design LSTM weights to compute a cumulative sentiment score over a document containing the words "bad", "good", "not", and "uh", where the score is:

**score = Σ"good" - Σ"bad" - 2·Σ"not good" + 2·Σ"not bad"**

Words are one-hot encoded in alphabetical order: bad=[1,0,0,0], good=[0,1,0,0], not=[0,0,1,0], uh=[0,0,0,1].

**Architecture:** input size=4, hidden size=1, 1 layer

**Key weight choices:**

The input gate weights Wi = [-1, 1, -2, 0] assign the raw sentiment contribution of each word: bad contributes -1, good +1, not -2 (anticipating a sign flip for the following word), uh 0.

The output gate bias bo=1 keeps the output gate open for all words except "not", which uses bo=0, preventing "not" itself from contributing to the score while its effect is stored in the cell state for the next word.

**Worked example: "good not bad bad"**

| Step | Word | Cell state | Hidden state | Output |
|---|---|---|---|---|
| 1 | good | c1 = 1 | h1 = 1 | +1 |
| 2 | not | c2 = -2 | h2 = 0 | 0 (gate closed) |
| 3 | bad | c3 = (-1)(-2) = 2 | h3 = 2 | +2 |
| 4 | bad | c4 = -1 | h4 = -1 | -1 |

Final score: 1 + 0 + 2 - 1 = 2, which matches the formula: 1 "good" - 1 "bad" + 2·(1 "not bad") = 1 - 1 + 2 = 2.