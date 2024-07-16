import torch

words = open("./data/names.txt", "r").read().splitlines()

freq_dict = {}
for w in words:
    chrs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chrs, chrs[1:]):
        bigram = (ch1, ch2)
        freq_dict[bigram] = freq_dict.get(bigram, 0) + 1

s = sorted(freq_dict.items(), key = lambda kv: -kv[1])
print(s)

