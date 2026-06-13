import numpy as np
from nltk.corpus import brown
from collections import defaultdict, Counter
from typing import Sequence, Tuple, TypeVar

Q = TypeVar("Q")
V = TypeVar("V")


def prepare_data():
    tagged_sentences = brown.tagged_sents(tagset="universal")[:10000]

    tags = []
    words = []

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

    transition_matrix = np.ones((n_tags, n_tags))
    observation_matrix = np.ones((n_tags, n_words + 1))  # +1 for UNK

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

    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    observation_matrix /= observation_matrix.sum(axis=1, keepdims=True)

    initial_state_distribution = np.ones(n_tags)
    for sent in tagged_sentences:
        initial_tag = tag_to_idx[sent[0][1]]
        initial_state_distribution[initial_tag] += 1
    initial_state_distribution /= initial_state_distribution.sum()

    return (
        tag_to_idx,
        word_to_idx,
        transition_matrix,
        observation_matrix,
        initial_state_distribution,
    )


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[Q], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]
    print(qs)
    return qs, np.exp(log_ps)


def tag_sentences(
    test_sentences,
    tag_to_idx,
    word_to_idx,
    initial_state_distribution,
    transition_matrix,
    observation_matrix,
):

    predicted_tags = []

    for sent in test_sentences:
        obs = []
        for word, _ in sent:
            obs.append(word_to_idx.get(word, -1))
        pred, _ = viterbi(
            obs, initial_state_distribution, transition_matrix, observation_matrix
        )
        predicted_tags.append(pred)
    return predicted_tags


if __name__ == "__main__":
    (
        tag_to_idx,
        word_to_idx,
        transition_matrix,
        observation_matrix,
        initial_state_distribution,
    ) = prepare_data()
    test_sentences = brown.tagged_sents(tagset="universal")[10150:10153]
    predicted_tags = tag_sentences(
        test_sentences,
        tag_to_idx,
        word_to_idx,
        initial_state_distribution,
        transition_matrix,
        observation_matrix,
    )

    # Assuming `predicted_tags` is a list of dictionaries returned from the Viterbi function
    for sent_idx, pred in enumerate(predicted_tags):
        print(f"Sentence {10150 + sent_idx}:")
        for word_idx in range(len(test_sentences[sent_idx])):
            word = test_sentences[sent_idx][word_idx][0]
            tag_index = pred[word_idx]

            tag = tag_to_idx.get(tag_index, "UNKNOWN")

            if tag is None:
                tag = "UNKNOWN"

            print(f"Word: {word}, Tag Index: {tag_index}, Tag: {tag}")

            if word in [".", "!", "?", ","]:
                tag = "PUNCTUATION"

            print(f"{word}/{tag}", end=" ")
        print()  # New line after each sentence
