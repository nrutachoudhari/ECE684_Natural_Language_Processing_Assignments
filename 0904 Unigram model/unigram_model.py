import random


def predict_word():
    """Predict a word."""
    return "the"


def predict_unigram():
    """Predict a word."""
    return {"the": 0.6, "of": 0.3, "dinosaur": 0.1}


def sample_from_distribution(dist: dict[str, float], k: int):
    """Sample from distribution."""
    population = list(dist.keys())
    weights = list(dist.values())
    print(population, weights)
    return random.choices(population, weights, k=k)


if __name__ == "__main__":
    print(predict_unigram())
    print(sample_from_distribution(predict_unigram(), 20))
