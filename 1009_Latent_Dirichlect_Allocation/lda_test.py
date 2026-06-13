"""
ECE 684 Natural Language Processing
Assignment 6 - Latent Dirichlet Allocation
Nruta Choudhari
"""

"""
Latent Dirichlet Allocation

Patrick Wang, 2021
"""

from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np
from collections import defaultdict


def lda_gen(
    vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int
) -> List[str]:
    # sample document length from a Poisson distribution
    doc_length = np.random.poisson(xi)

    # draw topic distribution for the document from a Dirichlet distribution
    topic_dist = np.random.dirichlet(alpha)

    # initialize an empty list to store words
    words = []

    # for each word in the document, choose a topic and then a word
    for _ in range(doc_length):
        topic = np.random.choice(len(topic_dist), p=topic_dist)

        word = np.random.choice(vocabulary, p=beta[topic])

        words.append(word)
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass",
        "pike",
        "deep",
        "tuba",
        "horn",
        "catapult",
    ]
    beta = np.array(
        [
            [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
            [0.3, 0.0, 0.2, 0.3, 0.2, 0.0],
        ]
    )
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [lda_gen(vocabulary, alpha, beta, xi) for _ in range(100)]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())


if __name__ == "__main__":
    test()
