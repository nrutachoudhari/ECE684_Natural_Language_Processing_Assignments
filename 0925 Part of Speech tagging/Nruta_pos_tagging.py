"""
ECE 684 Natural Language Processing
Assignment 4 - Part of Speech Tagging
Nruta Choudhari
"""

import numpy as np
from nltk.corpus import brown
from typing import Sequence, Tuple, TypeVar
from viterbi import viterbi

Q = TypeVar("Q")
V = TypeVar("V")

# loading the first 10k tagged sentences
tagged_sentences = brown.tagged_sents(tagset="universal")[:10000]

tags = []
words = []

# creating the mappings
for sent in tagged_sentences:
    for _, tag in sent:
        tags.append(tag)

for sent in tagged_sentences:
    for word, _ in sent:
        words.append(word)

tag_to_idx = {}
word_to_idx = {}

for idx, tag in enumerate(set(tags)):
    tag_to_idx[tag] = idx

for idx, word in enumerate(set(words)):
    word_to_idx[word] = idx

n_tags = len(tag_to_idx)
n_words = len(word_to_idx)

# initializing the matrices with -1 smoothing everywhere
transition_matrix = np.ones((n_tags, n_tags))
observation_matrix = np.ones((n_tags, n_words + 1))  # +1 for UNK

# fill the matrices based on tagged sentences
for sent in tagged_sentences:
    for i in range(1, len(sent)):
        prev_tag = tag_to_idx[sent[i - 1][1]]
        curr_tag = tag_to_idx[sent[i][1]]
        transition_matrix[prev_tag, curr_tag] += 1

        word = sent[i][0]
        if word in word_to_idx:
            observation_matrix[curr_tag, word_to_idx[word]] += 1
        else:
            observation_matrix[curr_tag, -1] += 1  # UNK

# normalize to get the probabilities
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
observation_matrix /= observation_matrix.sum(axis=1, keepdims=True)

initial_state_distribution = np.ones(n_tags)
for sent in tagged_sentences:
    initial_tag = tag_to_idx[sent[0][1]]
    initial_state_distribution[initial_tag] += 1
initial_state_distribution /= initial_state_distribution.sum()

# loading the test sentences
test_sentences = brown.tagged_sents(tagset="universal")[10150:10153]
predicted_tags = []

for sent in test_sentences:
    obs = []
    for word, _ in sent:
        obs.append(word_to_idx.get(word, -1))
    pred, _ = viterbi(
        obs, initial_state_distribution, transition_matrix, observation_matrix
    )
    predicted_tags.append(pred)


# comparing the taggings received with the true taggings of the words in the sentences
def compare_predictions_with_truth(predicted_tags, test_sentences, start_idx=10150):
    correct = 0
    total = 0

    # generating the word component of the sentence only
    for sent_idx, pred in enumerate(predicted_tags):
        sentence_only = []
        for word, _ in test_sentences[sent_idx]:
            sentence_only.append(word)

        print(f"Sentence {start_idx + sent_idx}")
        print(sentence_only)
        print("POS Tagging:")
        for word_idx in range(len(test_sentences[sent_idx])):
            true_word, true_tag = test_sentences[sent_idx][word_idx]
            predicted_tag_index = pred[word_idx]

            predicted_tag = None
            for k, v in tag_to_idx.items():
                if v == predicted_tag_index:
                    predicted_tag = k
                    break

            print(
                f"Word: {true_word}, True: {true_tag}, Predicted: {predicted_tag} |",
                end=" ",
            )
            print("\n")

            # checking if the tag predicted is the same as the actual
            if predicted_tag == true_tag:
                correct += 1

            total += 1

        print("\n")

    if total > 0:
        accuracy = correct / total
    else:
        0
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy


compare_predictions_with_truth(predicted_tags, test_sentences)
