"""
Markov Text Generator.

Nruta Choudhari, 2024

Assignment 2
"""

from collections import defaultdict
import numpy as np
import random


def build_ngram_model(corpus, n):
    # builds an n-gram model from the given corpus

    counts = defaultdict(lambda: defaultdict(lambda: 0.0))
    sentences = tuple(corpus)
    for i in range(len(sentences)):
        for k in range(n):
            if i - k < 0:
                break
            counts[sentences[i - k : i]][sentences[i]] += 1

    return counts


def calculate_probabilities(counts):
    # calculates the probabilities of words given their context
    prob_ngram = defaultdict(lambda: defaultdict(lambda: 0.0))
    for context in counts.keys():
        denom = sum(counts[context].values())
        if denom > 0:
            for word in counts[context].keys():
                prob_ngram[context][word] = counts[context][word] / denom
    return prob_ngram


def stupid_backoff_func(model, context):
    # applying stupid backoff to find the word probabilities
    if context in model:
        return model[context]
    else:
        # Apply backoff: multiply probabilities by 0.4 and use shorter context
        return {
            word: 0.4 * prob
            for word, prob in stupid_backoff_func(model, context[1:]).items()
        }


def finish_sentence(sentence, n, corpus, randomize=False):
    # generates a sentence based on the input sentence and using the Markov model and backoff strategy

    counts = build_ngram_model(corpus, n)  # Build the n-gram model
    probabilities = calculate_probabilities(counts)  # Calculate word probabilities
    current_sentence = list(sentence)

    while len(current_sentence) < 10:
        context = tuple(current_sentence[-n:])  # Get the last n words for context
        next_words = stupid_backoff_func(probabilities, context)  # Apply stupid backoff

        if next_words:
            if randomize:
                # randomize = true, so it chooses randomly
                next_word = random.choices(
                    list(next_words.keys()), list(next_words.values())
                )[0]
            else:
                # randomize = False, so it chooses the next word with the highest probability. if there are ties, chooses alphabetically amongst the ties
                next_word = sorted(next_words.items(), key=lambda x: (-x[1], x[0]))[0][
                    0
                ]
        else:
            break

        current_sentence.append(next_word)

        # Stop if a sentence-ending punctuation is found
        if next_word in [".", "?", "!"]:
            break

    return current_sentence
