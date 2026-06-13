# Noisy Channel Model: Spelling Correction

## Overview

This assignment implements a spelling corrector using the noisy channel model, combining a unigram language model with a weighted Levenshtein distance error model. Given a misspelled word, the corrector generates all single-edit candidates, scores them, and returns the most probable correction.

## Files

- `main.py` — final implementation with `correct(original: str) -> str`
- `edits.py` — edit generation logic (deletions, substitutions, additions)
- `unigrams.csv` — word unigram frequencies
- `bigrams.csv` — word bigram frequencies
- `additions.csv`, `deletions.csv`, `substitutions.csv` — confusion matrix data for error probabilities
- `count_1w.txt` — Peter Norvig's word frequency corpus used as the language model
- `Noisy_Channel_Model.pdf` — written report covering assumptions, test cases, and failure analysis
- `debug.py`, `rough.py`, `actual_debug.py` — development and debugging scripts

## How It Works

`correct(original)` generates all real-word candidates reachable by a single edit (deletion, insertion, or substitution) that appear in the corpus, then scores each candidate as:

```
score = P(candidate) * channel_probability(edit)
```

The channel probability is computed using confusion matrices: for each edit type, the count of that specific error divided by the frequency of the correct character (or bigram context for deletions). The highest-scoring candidate is returned.

## Modeling Assumptions

1. **Single error per word:** each misspelled word contains at most one error (edit distance of 1)
2. **No transpositions:** the model handles deletions, insertions, and substitutions only
3. **Vocabulary restricted to corpus:** words not in `count_1w.txt` are not considered valid candidates

## Example Outputs

| Input | Corrected | Notes |
|---|---|---|
| helo | hello | Correct |
| definately | definitely | Correct |
| tee | tree | Correct |
| tye | the | Correct |
| recieve | recieved | Incorrect: transposition not modeled |
| quizdacious | quizdacious | Incorrect: rare word not in corpus |

## Known Limitations

Transposition errors (swapping adjacent characters like `ie` to `ei`) are not handled since the edit model only considers deletions, insertions, and substitutions. Words outside the Norvig corpus have no valid candidates and are returned unchanged. Rare but correctly spelled words can also be outscored by more frequent incorrect alternatives.

## Libraries Used

`csv`, `collections`, `math`

## References

Jurafsky and Martin, *Speech and Language Processing*, 2024, Appendix B