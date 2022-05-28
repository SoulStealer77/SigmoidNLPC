import copy
import re
import unicodedata
import pandas as pd

import torch
torch.manual_seed(17)
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

import matplotlib.pyplot as plt



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

PAD_token = 0

class Vocabulary:
    def __init__(self):
        self.PAD_TOKEN = 0
        self.word2index = {}
        self.word2count = {}
        self.index2word = { self.PAD_TOKEN: 'PAD' }
        self.words_count = 1
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.words_count
            self.word2count[word] = 1
            self.index2word[self.words_count] = word
            self.words_count += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence = normalizeString(sentence)
        tokens = self.tokenizer(sentence)

        if len(tokens) > self.max_length:
            self.max_length = len(tokens)

        for token in tokens:
            self.add_word(token)

    def sentence2indices(self, sentence):
        sentence = normalizeString(sentence)
        result = [ self.PAD_TOKEN ] * self.max_length
        idx = 0
        for token in self.tokenizer(sentence):
            if token in self.word2index:
                result[idx] = self.word2index[token]
                idx += 1

        return result

vocabulary = Vocabulary()


df_train = pd.read_csv('twitter_training.csv', usecols=[2, 3], names=['sentiment','text'])
df_train.head()

df_train.isna().sum()

df_train.drop(df_train.loc[df_train['text'].isna()].index, inplace=True)
df_train.reset_index(inplace=True)




for text in df_train['text']:
  vocabulary.add_sentence(text)


class GRU(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, num_words, drop_prob=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_words, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.gru(x)
        out = self.fc(self.ReLU(out[:, -1]))
        return out


model = GRU(num_words=vocabulary.words_count, hidden_dim=128, output_dim=3, n_layers=2)
model.load_state_dict(torch.load((r'C:\Users\grigo\OneDrive\Desktop\nlpcv2\model_best.pt'),map_location=torch.device('cpu')))
model.eval()

message = 'We lost connection'
message_indices = vocabulary.sentence2indices(message)
print(message_indices)
result = model(torch.tensor(message_indices).view(1,-1))
index_sentiment = torch.argmax(result)
index_to_sentiment = {0:'Positive',1 : 'Neutral', 2: 'Negative'}
print(index_to_sentiment[index_sentiment.item()])


