import csv


def load_unigrams(filepath: str) -> dict[str, int]:
    unigrams = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            word, freq = row
            unigrams[word] = int(freq)
    return unigrams


def load_bigrams(filepath: str) -> dict[tuple[str, str], int]:
    bigrams = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            combo, freq = row
            bigrams[tuple(combo.split())] = int(freq)
    return bigrams


def load_additions(filepath: str) -> dict[str, list[tuple[str, int]]]:
    additions = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            prefix, added, count = row
            count = int(count)
            additions.setdefault(prefix, []).append((added, count))
    return additions


def load_deletions(filepath: str) -> dict[str, list[tuple[str, int]]]:
    deletions = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            prefix, deleted, count = row
            count = int(count)
            deletions.setdefault(prefix, []).append((deleted, count))
    return deletions


def load_substitutions(filepath: str) -> dict[str, list[tuple[str, int]]]:
    substitutions = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            original, substituted, count = row
            count = int(count)
            substitutions.setdefault(original, []).append((substituted, count))
    return substitutions


def get_single_edit_edits(
    original: str, characters: list[str]
) -> list[tuple[str, str]]:
    edits = []

    # Generate deletions
    for idx in range(len(original)):
        edits.append((f"d:{original[idx]}", original[:idx] + original[idx + 1 :]))

    # Generate substitutions
    for idx, old_char in enumerate(original):
        for new_char in characters:
            if new_char != old_char:
                edits.append(
                    (
                        f"s:{old_char}{new_char}",
                        original[:idx] + new_char + original[idx + 1 :],
                    )
                )

    # Generate additions
    for idx in range(len(original) + 1):
        for new_char in characters:
            edits.append((f"a:{new_char}", original[:idx] + new_char + original[idx:]))

    return edits


def correct(
    original: str, unigrams, bigrams, additions, deletions, substitutions
) -> str:
    characters = "abcdefghijklmnopqrstuvwxyz"
    edits = get_single_edit_edits(original, characters)

    best_correction = original
    highest_probability = 0

    for edit_type, candidate in edits:
        probability = calculate_probability(
            edit_type,
            original,
            candidate,
            unigrams,
            bigrams,
            deletions,
            additions,
            substitutions,
        )

        if probability > highest_probability:
            highest_probability = probability
            best_correction = candidate
        print(
            f"Edit Type: {edit_type}, Candidate: {candidate}, Probability: {probability}"
        )

    return best_correction


def calculate_probability(
    edit_type: str,
    original: str,
    candidate: str,
    unigrams: dict[str, int],
    bigrams: dict[tuple[str, str], int],
    deletions,
    additions,
    substitutions,
) -> float:
    unigram_prob = get_unigram_probability(candidate, unigrams)
    error_prob = calculate_error_probability(
        edit_type, original, candidate, bigrams, deletions, additions, substitutions
    )

    return 0.7 * unigram_prob + 0.3 * error_prob


def calculate_error_probability(
    edit_type: str,
    original: str,
    candidate: str,
    bigrams,
    deletions,
    additions,
    substitutions,
) -> float:
    if edit_type.startswith("d:"):  # Deletion
        for deleted, count in deletions.get(original[0], []):
            if original.replace(deleted, "", 1) == candidate:
                return count / sum(cnt for _, cnt in deletions.get(original[0], []))

    elif edit_type.startswith("a:"):  # Addition
        for added, count in additions.get(original[0], []):
            if candidate == original + added:
                return count / sum(cnt for _, cnt in additions.get(original[0], []))

    elif edit_type.startswith("s:"):  # Substitution
        for old_char, new_char in zip(original, candidate):
            if old_char != new_char:
                for substituted, count in substitutions.get(old_char, []):
                    if substituted == new_char:
                        return count / sum(
                            cnt for _, cnt in substitutions.get(old_char, [])
                        )

    return 0.01  # Fallback probability


def get_unigram_probability(word: str, unigrams: dict[str, int]) -> float:
    total_count = sum(unigrams.values())
    word_count = unigrams.get(word, 0)
    return word_count / total_count if total_count > 0 else 0.0


if __name__ == "__main__":
    unigrams = load_unigrams("unigrams.csv")
    bigrams = load_bigrams("bigrams.csv")
    additions = load_additions("additions.csv")
    deletions = load_deletions("deletions.csv")
    substitutions = load_substitutions("substitutions.csv")

    test_words = ["helo", "definately", "writting", "recieve", "occured"]
    for word in test_words:
        corrected = correct(
            word, unigrams, bigrams, additions, deletions, substitutions
        )
        print(f"Input: {word}, Corrected: {corrected}")
