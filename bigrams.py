import torch
from utils.constants import special_char
from typing import TypeVar

data = open("./data/names.txt", "r").read().splitlines()
T = TypeVar('T')

# return a list of all unique characters found in data
def initialize_vocab(data):
        return sorted(list(set("".join(data))) + [special_char])

# Create mapping of bigram frequencies to frequency count from given data
def create_bigram_freq_dict():
    # Bigram implementation (first step; change to transformer impl. later)
    bigram_frequencies = {}
    for w in data:
        chrs = [special_char] + list(w) + [special_char]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            bigram = (ch1, ch2)
            bigram_frequencies[bigram] = bigram_frequencies.get(bigram, 0) + 1

        return sorted(bigram_frequencies.items(), key = lambda kv: -kv[1])

# Create a 2D tensor where rows = start char & col = next char (char that comes next in sequence)
# NOTE: Can change zeros -> ones to add 'fake counts', or adjust in probability P directly
def create_bigram_freq_tensor():
    freq_2d = torch.zeros((len(vocab), len(vocab)), dtype = torch.int32)

    for w in data:
        chrs = [special_char] + list(w) + [special_char]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            freq_2d[ix1, ix2] += 1
    
    return freq_2d

# Create a columnn vector of probabilities from the normalized values from data
# NOTE: Broadcasting semantics @ PyTorch about how tensors of unequal length can still perform element-wise op.
def create_probabilities(vocab_frequencies: torch.Tensor):
    # convert freq array into list of probabilities and sum into column vector
    P = vocab_frequencies.float() # NOTE: Do P = (vocab_frequencies + <fake_weight_count>) here to avoid -inf -> bigger = more uniform, less = more peaked
    P /= P.sum(1, keepdim = True)

    return P

# Generate names
def sample_model(P):
    g = torch.Generator().manual_seed(2147483647)
    results = []

    # Generate 10 names
    for i in range(10):
        output = ""
        idx = 0

        # Generate a name
        while True:
            idx = torch.multinomial(P[idx], num_samples = 1, replacement = True, generator = g).item()

            if (idx == 0):
                results.append(output)
                break

            output += itos[idx]
        
    return results

def calculate_loss(data):
    # GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
    # equivalent to maximizing the log likelihood (because log is monotonic)
    # equivalent to minimizing the negative log likelihood
    # equivalent to minimizing the average negative log likelihood

    # log(a*b*c) = log(a) + log(b) + log(c)

    log_likelihood = 0.0
    n = 0

    for word in data[:5]:
        chrs = [special_char] + list(word) + [special_char]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

        print(f'log_likelihood: {log_likelihood=}')
        nll = -log_likelihood
        print(f'{nll=}')
        print(f'{nll/n}')

vocab = initialize_vocab(data)
stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in enumerate(vocab)}
vocab_frequencies = create_bigram_freq_tensor()
P = create_probabilities(vocab_frequencies)

sample = sample_model(P)

calculate_loss(data)