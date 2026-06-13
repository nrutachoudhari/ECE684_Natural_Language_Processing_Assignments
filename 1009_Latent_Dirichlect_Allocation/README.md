# Latent Dirichlet Allocation

## Overview

This assignment implements an LDA document generator from scratch and validates it by running gensim's LDA inference on the generated corpus, then comparing the inferred topic-word distributions against the known ground truth.

## Files

- `lda_test.py` — implements `lda_gen()` and runs inference using gensim's `LdaModel`
- `Latent_Dirichlet_Allocation.pdf` — written analysis mapping inferred topics to true topics
- `lda.pdf` — assignment specification

## How It Works

`lda_gen(vocabulary, alpha, beta, xi)` generates a single document by:

1. Drawing document length from `Poisson(xi)`
2. Drawing a topic distribution for the document from `Dirichlet(alpha)`
3. For each word position, sampling a topic from that distribution, then sampling a word from the corresponding row of `beta`

100 documents are generated and fed into gensim's `LdaModel` with 3 topics for inference.

## Vocabulary and True Topics

The vocabulary is 6 ambiguous words — "bass", "pike", "deep", "tuba", "horn", "catapult" — chosen specifically because they span multiple semantic fields, making topic recovery non-trivial.

| True Topic | High-weight Words | Theme |
|---|---|---|
| Topic 1 | bass (0.4), pike (0.4), deep (0.2) | Fishing |
| Topic 2 | pike (0.3), horn (0.3), catapult (0.3) | Hunting/weapons |
| Topic 3 | bass (0.3), deep (0.2), tuba (0.3) | Music |

## Inferred vs. True Topics

| Inferred Topic | Top Words | Maps To |
|---|---|---|
| Topic 0 | tuba (0.279), horn (0.224), bass (0.218) | True Topic 3 (Music) |
| Topic 1 | bass (0.341), pike (0.230), deep (0.187) | True Topic 1 (Fishing) |
| Topic 2 | pike (0.358), horn (0.204), catapult (0.175) | True Topic 2 (Hunting) |

The inferred topics recover the true structure reasonably well. Some noise is expected since the vocabulary words are deliberately ambiguous across topics (e.g., "bass" appears in both fishing and music contexts, "horn" in both music and hunting).

## Libraries Used

`gensim`, `numpy`