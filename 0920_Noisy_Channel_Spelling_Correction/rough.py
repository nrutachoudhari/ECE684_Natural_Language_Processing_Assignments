import csv
from collections import defaultdict


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


"""def calculate_probabilities(
    word, unigrams, bigrams, additions, deletions, substitutions, corpus
):
    
    Calculate correction candidates and their probabilities using the noisy channel model.
    
    candidates = {}

    # Get possible edits for the original word
    characters = unigrams.keys()
    edits = get_edits(word, characters, corpus)

    total_unigram = sum(unigrams.values())
    total_bigram = sum(bigrams.values())

    # Calculate probabilities for each edit
    for edit_type, edited_word in edits:
        if edit_type.startswith("a:"):  # Handle additions
            if edited_word in unigrams:
                prob = unigrams[edited_word] / total_unigram
                candidates.append((prob, edited_word))
        elif edit_type.startswith("d:"):  # Handle deletions using bigrams
            prefix = edit_type[2:]
            if (prefix, edited_word) in bigrams:
                prob = bigrams[(prefix, edited_word)] / total_bigram
                candidates.append((prob, edited_word))
        elif edit_type.startswith("s:"):  # Handle substitutions
            if edited_word in unigrams:
                prob = unigrams[edited_word] / total_unigram
                candidates.append((prob, edited_word))

    return candidates"""


def weighted_levenshtein(
    unigrams, bigrams, substitutions, additions, deletions, word, corpus
):
    candidates = {}
    characters = unigrams.keys()

    # Get possible edits for the original word
    edit = get_edits(word, characters, corpus)

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


def correct(original: str) -> str:
    candidates = weighted_levenshtein(
        unigrams, bigrams, additions, deletions, substitutions, original, corpus
    )

    if not candidates:
        return original

    best_candidate = max(candidates, key=lambda x: x[0])

    return best_candidate[1]


if __name__ == "__main__":
    unigrams = load_unigrams("unigrams.csv")
    bigrams = load_bigrams("bigrams.csv")
    additions = load_additions("additions.csv")
    deletions = load_deletions("deletions.csv")
    substitutions = load_substitutions("substitutions.csv")
    corpus = load_this_csv("count_1w.txt")

    test_words = ["helo", "definately", "writting", "recieve", "occured"]
    for word in test_words:
        corrected = correct(word)
        print(f"Input: {word}, Corrected: {corrected}")
