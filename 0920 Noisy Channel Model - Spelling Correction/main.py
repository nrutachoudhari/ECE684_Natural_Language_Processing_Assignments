"""
ECE 684 Natural Language Processing
Assignment 3 - Noisy Channel Model
Nruta Choudhari
"""

import csv
from collections import defaultdict
from math import inf


def read_txt_file(filepath: str) -> list[str]:
    with open(filepath, mode="r", encoding="utf-8") as file:  # Open the file
        lines = file.readlines()  # Read all lines into a list
    return [line.strip() for line in lines]  # Strip newlines and return the list


def load_unigrams(filepath: str) -> dict[str, int]:
    unigrams = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            word, freq = row
            unigrams[word] = int(freq)
    return unigrams


def load_bigrams(filepath: str) -> dict[str, int]:
    bigrams = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            combo, freq = row
            bigrams[combo] = int(freq)
    return bigrams


def load_additions(filepath: str) -> dict[str, dict[str, int]]:
    additions = defaultdict(dict)
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            prefix, added, count = row
            additions[prefix][added] = int(count)
    return additions


def load_deletions(filepath: str) -> dict[str, dict[str, int]]:
    deletions = defaultdict(dict)
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            prefix, deleted, count = row
            deletions[prefix][deleted] = int(count)
    return deletions


def load_substitutions(filepath: str) -> dict[str, dict[str, int]]:
    substitutions = defaultdict(dict)
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            original, substituted, count = row
            substitutions[original][substituted] = int(count)
    return substitutions


def load_this_csv(filepath: str) -> dict[str, int]:
    corpus = {}
    with open(filepath, mode="r") as file:
        for row in file:
            word, count = row.split()
            corpus[word] = int(count)
    return corpus


def word_prob(corpus):
    total_num = sum(corpus.values())
    word_probabilities = defaultdict(float)
    for word, frequency in corpus.items():
        probability = frequency / total_num
        word_probabilities[word] = probability
    return word_probabilities


def get_edits(original, characters, count_1w):
    edits = defaultdict(lambda: defaultdict(str))

    # Generate additions
    for idx, char in enumerate(original):
        previous_char = original[idx - 1] if idx > 0 else "#"
        if count_1w.get(original[:idx] + original[idx + 1 :]):
            edits[original[:idx] + original[idx + 1 :]]["a"] = previous_char + char

    # Generate substitutions
    for idx, old_char in enumerate(original):
        for new_char in characters:
            if count_1w.get(original[:idx] + new_char + original[idx + 1 :]):
                edits[original[:idx] + new_char + original[idx + 1 :]]["s"] = (
                    old_char + new_char
                )

    # Generate deletions
    for idx, char in enumerate("#" + original):
        for new_char in characters:
            if count_1w.get(original[:idx] + new_char + original[idx:]):
                edits[original[:idx] + new_char + original[idx:]]["d"] = char + new_char

    return edits


def weighted_levenshtein(edit, unigrams, bigrams, substitutions, additions, deletions):
    cost = 0
    for key, value in edit.items():
        if key == "d":
            deletion = deletions.get(value[0], {}).get(value[1], 0) / (
                bigrams.get(value, inf)
                if value[0] != "#"
                else unigrams.get(value[1], inf)
            )
            cost += deletion
        elif key == "a":
            addition = additions.get(value[0], {}).get(value[1], 0) / unigrams.get(
                value[0], inf
            )
            cost += addition
        elif key == "s":
            substitution = substitutions.get(value[0], {}).get(
                value[1], 0
            ) / unigrams.get(value[1], inf)
            cost += substitution

    return cost


def correct(original):
    unigrams = load_unigrams("unigrams.csv")
    bigrams = load_bigrams("bigrams.csv")
    additions = load_additions("additions.csv")
    deletions = load_deletions("deletions.csv")
    substitutions = load_substitutions("substitutions.csv")
    corpus = load_this_csv("count_1w.txt")

    prob_of_word = word_prob(corpus)
    edits = get_edits(original, unigrams.keys(), corpus)
    candidates = {}
    for edited_word, edit in edits.items():
        cost = weighted_levenshtein(
            edit, unigrams, bigrams, substitutions, additions, deletions
        )

        candidates[edited_word] = cost * prob_of_word.get(edited_word)

    if not candidates:
        return original

    best_candidate = max(candidates, key=candidates.get)

    return best_candidate


# test cases
print("Cases where the Noisy Channel model should work as expected")
print(f"input word: helo, corrected word: {correct("helo")}")
print(f"input word: definately, corrected word: {correct("definately")}")
print(f"input word: tee, corrected word: {correct("tee")}")
print(f"input word: tye, corrected word: {correct("tye")}")

print("-----------------------------------------------------------")
# test cases which give the wrong output
print("Cases where the Noisy Channel Model did not work as expected")
print(f"input word: recieve, corrected word: {correct("recieve")}") # the output should be "receive" but transposition is not considered in this model
print(f"input word: quizdacious, corrected word: {correct("quizdacious")}") # rare words that don't occue often. the correct spelling should be "quizzacious"
