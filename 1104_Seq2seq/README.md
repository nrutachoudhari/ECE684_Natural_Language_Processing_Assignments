# Seq2Seq: RNN Encoder-Decoder

## Overview

This is a theoretical assignment requiring manual weight design for a simple RNN encoder-decoder that summarizes token sequences into count vectors. No code or training was required.

## File

- `Nruta_Choudhari_NLP_Assignment_9.pdf` — handwritten solution with matrix derivations

---

## The Task

Given documents composed of tokens {a, b, c, d, e} plus an EOS token (.), the model must output a sequence of length 5 representing the count of each token in the input.

Examples:
- "badcab." → "22110."
- "bababacee." → "33101."
- "dadda." → "20030."

---

## Architecture

The model is a simple RNN encoder-decoder:

```
encode(x, h) = We * [x; h]
decode(h)    = ReLU(Wo * h),  Wh * h
```

**Sizes:**
- Input x: 6-dimensional one-hot vector (tokens a, b, c, d, e + EOS)
- Hidden state h: 6-dimensional (one dimension per token to accumulate counts, plus one for EOS detection)
- Output: 2-dimensional (scalar count + EOS indicator)

---

## Weight Design

**We (6x12):** The encoder weight matrix concatenates the input and hidden state to produce a new hidden state. Each row is designed so that the hidden state simply accumulates counts by adding 1 to the appropriate dimension whenever a token is seen. Entries are 0s, 1s, and a -1 to flag the EOS condition. The EOS row uses -1 to distinguish the end-of-sequence token from regular tokens.

**Whh (6x6):** The hidden-to-hidden transition matrix is a 6x6 identity-like structure with 0s and 1s that carries the accumulated counts forward at each time step without modification. The straightforward 0/1 entries reflect that the hidden state only needs to be passed through unchanged between steps.

**Wo (2x6):** The output matrix has dimension 2x6 and extracts the relevant count and EOS signal from the hidden state. A -1 entry flags the EOS condition in the hidden state, which is used to trigger the end-of-sequence output indicator. The decoder applies ReLU to ensure counts are non-negative.

---

## Key Insight

Because the task is purely counting with no ordering dependency, the encoder only needs to accumulate token occurrences additively into the hidden state. The hidden state acts as a running tally, and the decoder simply reads off the final counts after the EOS token is processed. This makes the weight design tractable by hand since no nonlinear interaction between tokens is needed.