# ECE 684: Natural Language Processing — Fall 2024

This repository contains my work from **Natural Language Processing (ECE 684)** at Duke University's AI for Product Innovation program. The course builds up NLP from first principles, starting with probabilistic language models and progressing through neural sequence models.

Each assignment involves either implementing a core NLP algorithm from scratch or deriving and designing model weights by hand, with a strong emphasis on understanding the math behind the methods rather than using off-the-shelf libraries.

---

## Assignments

| # | Topic | Format | Key Methods |
|---|---|---|---|
| [A1](./0904_Unigram_Model/) | Unigram Model | Python + written proof | MLE, Lagrange multipliers |
| [A2](./0911_Markov_Text_Generation/) | Markov Text Generator | Python | N-gram models, stupid backoff |
| [A3](./0920_Noisy_Channel_Spelling_Correction/) | Spelling Correction | Python + written report | Noisy channel model, weighted Levenshtein distance |
| [A4](./0925_Part_of_Speech_Tagging/) | Part-of-Speech Tagging | Python | HMM, Viterbi algorithm |
| [A5](./1003_Gradient_Descent/) | Gradient Descent | Python | PyTorch, character-level unigram LM |
| [A6](./1009_Latent_Dirichlet_Allocation/) | Latent Dirichlet Allocation | Python + written analysis | LDA, Dirichlet sampling, gensim |
| [A7](./1025_LSTM/) | LSTMs | Written derivation | RNN, LSTM gates, sentiment scoring |
| [A9](./1104_Seq2Seq/) | Seq2Seq Encoder-Decoder | Written derivation | RNN encoder-decoder, manual weight design |

---

## Highlights

**Markov Text Generator (A2):** Implemented an n-gram language model with stupid backoff (α=0.4) that extends a seed sentence token by token. Tested on Jane Austen's *Sense and Sensibility* with a deterministic mode (highest-probability next word, alphabetical tie-breaking) and a randomized mode. All outputs validated against a provided test suite.

**Noisy Channel Spelling Correction (A3):** Built a spelling corrector combining a unigram language model with a weighted Levenshtein error model using confusion matrices for deletions, insertions, and substitutions. Correctly handles common misspellings like "helo" → "hello" and "definately" → "definitely", and includes honest documentation of where the model fails (transpositions, rare words outside the corpus).

**Part-of-Speech Tagging (A4):** Estimated HMM transition, emission, and initial state distributions from the first 10,000 Brown corpus sentences with add-1 smoothing and UNK handling. Decoded using Viterbi (log-space) and analyzed specific errors on three held-out sentences, tracing each mistake back to training data distributions.

**Gradient Descent (A5):** Extended a provided PyTorch character-level unigram model to track and visualize training loss and learned vs. optimal token probabilities. Compared three learning rate and iteration configurations and identified lr=0.01 with 1000 iterations as the best trade-off between final loss and training time.

**LDA (A6):** Implemented `lda_gen()` to synthesize documents from a known topic-word distribution over an intentionally ambiguous vocabulary (bass, pike, tuba, horn, catapult). Applied gensim's LDA inference to the generated corpus and mapped each inferred topic back to its true theme (fishing, music, hunting).

**LSTM Weight Design (A7):** Two theoretical problems: proving that a simple RNN with sigmoid activation is a strict subset of the LSTM by showing how to set each gate (forget, input, output) to a constant 1, and manually designing LSTM weights to compute a cumulative sentiment score that handles negation ("not good" and "not bad") by gating the output for "not" tokens.

**Seq2Seq (A9):** Manually designed encoder weight We, hidden-to-hidden matrix Whh, and decoder weight Wo for an RNN that reads a sequence of characters and outputs token counts. The key insight is that counting requires no ordering dependency, so the hidden state can act as a simple running tally with identity-like transitions.

---

## Topics Covered

- Probabilistic language models: unigram, bigram, n-gram with backoff
- Spelling correction: noisy channel model, confusion matrices, edit distance
- Sequence labeling: hidden Markov models, Viterbi decoding
- Neural language models: gradient descent, PyTorch, log-likelihood optimization
- Topic modeling: Latent Dirichlet Allocation, Dirichlet sampling
- Sequence models: RNNs, LSTMs, encoder-decoder architectures
- Theoretical derivations: MLE proofs, manual weight design, RNN/LSTM equivalence

---

## Tools and Libraries

Python, PyTorch, numpy, nltk, gensim, matplotlib, collections

---

*Duke University | Master of Engineering in AI for Product Innovation | Fall 2024*