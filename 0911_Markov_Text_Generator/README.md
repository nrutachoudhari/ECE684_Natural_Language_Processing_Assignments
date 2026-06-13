# Markov Text Generator

## Overview

This assignment implements a bare-bones Markov text generator that extends a seed sentence using n-gram language modeling with stupid backoff. The generator supports both deterministic and stochastic modes and is tested against expected outputs from Jane Austen's *Sense and Sensibility*.

## Files

- `mtg.py` — core implementation: n-gram model builder, probability calculator, stupid backoff, and `finish_sentence`
- `test_mtg.py` — test script that validates outputs against `test_examples.csv` using the Gutenberg Austen corpus
- `test_examples.csv` — test cases with seed inputs, n values, and expected outputs
- `Markov_chain_examples_NC.py` — additional example runs across n=2, 3, and 4 with custom corpora
- `Nruta_Choudhari_Markov_Text_Generator_Examples.pdf` — screenshots of example outputs

## How It Works

`finish_sentence(sentence, n, corpus, randomize=False)` extends a seed sentence token by token until it hits a sentence-ending punctuation (`.`, `?`, `!`) or reaches 10 tokens total.

At each step it looks up the last n tokens as context, applies stupid backoff (α=0.4) to handle unseen n-grams by falling back to shorter contexts, then either picks the highest-probability next word (deterministic, alphabetical tie-breaking) or samples from the distribution (randomized).

## Running the Tests

```bash
python test_mtg.py
```

Requires `nltk` with the Gutenberg corpus downloaded:

```python
import nltk
nltk.download('gutenberg')
nltk.download('punkt')
```

## Example Output

Seed: `["she", "was", "not"]`, n=3, corpus: Austen's *Sense and Sensibility*, randomize=False

```
she was not in the world , and the two
```

## Libraries Used

`nltk`, `numpy`, `collections`, `random`