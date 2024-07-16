import torch
from utils.constants import special_char

words = open("./data/names.txt", "r").read().splitlines()

# return a list of all unique characters found in words
def initialize_vocab(data):
    return sorted(list(set("".join(data))) + [special_char])

vocab = initialize_vocab(words)
stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in enumerate(vocab)}

def bigram():
    # Bigram implementation (first step; change to transformer impl. later)
    bigram_frequencies = {}
    for w in words:
        chrs = [special_char] + list(w) + [special_char]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            bigram = (ch1, ch2)
            bigram_frequencies[bigram] = bigram_frequencies.get(bigram, 0) + 1

        return sorted(bigram_frequencies.items(), key = lambda kv: -kv[1])

def bigram_tensors():
    freq_2d = torch.zeros((len(vocab), len(vocab)), dtype = torch.int32)

    for w in words:
        chrs = [special_char] + list(w) + [special_char]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            freq_2d[ix1, ix2] += 1

    for row in freq_2d:
        print(row.numpy())

bigram_tensors()