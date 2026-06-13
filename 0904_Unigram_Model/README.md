# Unigram Model

## Overview

This assignment covers the unigram language model: deriving the optimal parameter estimates from first principles and implementing a basic sampler in Python.

## Files

- `unigram_model.py` — implements word prediction and sampling from a unigram distribution
- `Nruta_Choudhari_Unigram_Model_Assignment.pdf` — written proof that the MLE estimate is optimal
- `unigram_model.pdf` — assignment specification

## What's Covered

**Written proof:** Given a vocabulary V and observed word counts, the unigram model assigns probability p_k = n_k / Σn_k to each word. The proof shows this is the maximum likelihood estimate using Lagrange multipliers, with the constraint that all probabilities sum to 1.

**Implementation:** The Python script demonstrates predicting a word, returning a unigram probability distribution, and sampling k words from that distribution using `random.choices` with frequency-based weights.

## Libraries Used

`random`