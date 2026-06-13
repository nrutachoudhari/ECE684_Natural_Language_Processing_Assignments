# Part-of-Speech Tagging

## Overview

This assignment implements a Hidden Markov Model (HMM) for part-of-speech tagging, trained on the Brown corpus and decoded using the Viterbi algorithm. The model is evaluated on three held-out sentences with word-level analysis of correct and incorrect predictions.

## Files

- `Nruta_pos_tagging.py` — builds the HMM and runs Viterbi on test sentences
- `viterbi.py` — provided Viterbi implementation (log-space for numerical stability)
- `test.py` — alternative test script with data preparation and tagging pipeline
- `Nruta_Part-of-Speech_tagging.pdf` — written analysis comparing predictions to ground truth

## How It Works

The HMM has three components, all estimated from the first 10,000 tagged sentences of the Brown corpus using the universal tagset:

- **Transition matrix:** tag-to-tag transition probabilities, add-1 smoothed
- **Observation matrix:** emission probabilities for each word given a tag, with an UNK token for out-of-vocabulary words and add-1 smoothing
- **Initial state distribution:** probability of each tag appearing at the start of a sentence

Decoding uses the Viterbi algorithm in log-space, which finds the most probable tag sequence for a given sentence by tracking both the best probabilities and backpointers at each step.

## Corpus

**Training:** `brown.tagged_sents(tagset='universal')[:10000]`  
**Test:** `brown.tagged_sents(tagset='universal')[10150:10153]`

## Results

| Sentence | Overall | Notable Errors |
|---|---|---|
| 10150 | Mostly correct | "Those" tagged PRON, true tag DET |
| 10151 | Mostly correct | "face-to-face" tagged NOUN (true: ADJ), "another" tagged NOUN (true: DET) |
| 10152 | All correct | Standard sentence structure matching training patterns |

**Error analysis:**

"Those" is ambiguous between pronoun and determiner. Because it more frequently appears as a pronoun in the training data, the model defaults to PRON even in determiner contexts.

"face-to-face" is a hyphenated compound that can function as either adjective or noun. Without explicit handling of hyphenated tokens, the model treats it like other words it has seen in noun positions.

"another" is commonly followed by a noun, so the model picks up a NOUN tag based on its learned transition context, when the correct tag is DET.

Sentence 10152 was tagged perfectly, likely because its syntactic structure closely matches patterns well-represented in the training data.

## Libraries Used

`numpy`, `nltk`