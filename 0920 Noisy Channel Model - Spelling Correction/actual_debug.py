import csv


def read_txt_file(filepath: str) -> list[str]:
    with open(filepath, mode="r", encoding="utf-8") as file:  # Open the file
        lines = file.readlines()  # Read all lines into a list
    return [line.strip() for line in lines]  # Strip newlines and return the list


if __name__ == "__main__":
    """unigrams = load_unigrams("unigrams.csv")
    bigrams = load_bigrams("bigrams.csv")
    additions = load_additions("additions.csv")
    deletions = load_deletions("deletions.csv")
    substitutions = load_substitutions("substitutions.csv")"""

    """test_words = ["helo", "definately", "writting", "recieve", "occured"]
    for word in test_words:
        corrected = correct(word, unigrams, bigrams, additions, deletions, substitutions)
        print(f"Input: {word}, Corrected: {corrected}")"""
    # Test the function
    corpus = read_txt_file("count_1w.csv")
    print(corpus)  # Print the first 10 entries to check
