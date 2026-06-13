import nltk
import random
from mtg import finish_sentence

"""corpus_n2 = [
    "data", "analysis", "is", "important", "for", "making", "decisions", 
    "in", "business", "data", "can", "reveal", "trends", "and", 
    "patterns", "that", "help", "understand", "customer", "behavior",
    "data", "analysis", "can", "also", "help", "in", "predictive", 
    "modeling", "and", "forecasting"
]

initial_sentence = ["data", "analysis"]
n = 2

output1 = finish_sentence(initial_sentence, n, corpus_n2, randomize=False)
print("the output for the corpus when n=2 and randomize = False")
print(" ".join(output1))
print("\n")
output2 = finish_sentence(initial_sentence, n, corpus, randomize= True)
print("the output for the corpus when n=2 and randomize = True")
print(" ".join(output2))"""


"""corpus = [
    "in", "the", "heart", "of", "the", "city", "there",
    "is", "a", "small", "cafe", "where", "people",
    "gather", "to", "drink", "coffee", "and", "chat",
    "about", "life", "and", "share", "their", "stories",
    "it", "is", "a", "place", "of", "comfort"
]


initial_sentence = ["in", "the", "heart"]
n = 3

output1 = finish_sentence(initial_sentence, n, corpus, randomize=False)
print("the output for the corpus when n=3 and randomize = False")
print(" ".join(output1))
print("\n")
output2 = finish_sentence(initial_sentence, n, corpus, randomize= True)
print("the output for the corpus when n=3 and randomize = True")
print(" ".join(output2))"""

corpus = [
    "data",
    "science",
    "is",
    "a",
    "field",
    "that",
    "uses",
    "data",
    "data",
    "science",
    "methods",
    "to",
    "analyze",
    "data",
    "data",
    "science",
    "helps",
    "in",
    "making",
    "decisions",
    "based",
    "on",
    "data",
    "data",
    "science",
    "can",
    "reveal",
    "patterns",
    "patterns",
    "help",
    "in",
    "understanding",
    "the",
    "data",
    "data",
    "science",
    "is",
    "used",
    "to",
    "improve",
    "business",
    "strategies",
    "using",
    "data",
    "data",
    "science",
    "provides",
    "insights",
    "that",
    "can",
    "drive",
    "innovation",
    "data",
    "science",
    "is",
    "essential",
    "for",
    "making",
    "informed",
    "decisions",
    "across",
    "various",
    "industries",
]

initial_sentence = ["data", "science"]

n = 4

output1 = finish_sentence(initial_sentence, n, corpus, randomize=False)
print("the output for the corpus when n=4 and randomize = False")
print(" ".join(output1))
print("\n")
output2 = finish_sentence(initial_sentence, n, corpus, randomize=True)
print("the output for the corpus when n=4 and randomize = True")
print(" ".join(output2))
