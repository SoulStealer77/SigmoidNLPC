import random
from email import message
from multiprocessing.connection import Client
import discord
import os
from dotenv import load_dotenv
from neuralintents import GenericAssistant
from requests import Response
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import spacy
import requests
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from discord.ext import commands

import copy
import re
import unicodedata
import pandas as pd

import torch
torch.manual_seed(17)
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math





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




USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
MAX_LENGTH = 10
MIN_COUNT = 3

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
IDK_token = 3  # Unknown token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", IDK_token: "IDK"}
        self.num_words = 4  # Count SOS, EOS, PAD, IDK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", IDK_token: "IDK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)
            
MIN_COUNT = 3    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

            
            
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
    
    
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
    
# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
    
    
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evalueate_input(input_sentence, encoder, decoder, searcher, voc):
    input_sentence = normalizeString(input_sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    return output_words


hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
attn_model = 'dot'

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ') if word in voc.word2index.keys()] + [EOS_token]  #[SOS_token] + 


def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

save_dir = os.path.join("data", "save")
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

embedding = nn.Embedding(voc.num_words, hidden_size).to(device)
embedding.load_state_dict(torch.load(r'C:\Users\grigo\OneDrive\Desktop\nlpcv2\embedding.pth',map_location=torch.device('cpu')))
embedding.eval()

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout).to(device)
encoder.load_state_dict(torch.load(r'C:\Users\grigo\OneDrive\Desktop\nlpcv2\encoder.pth',map_location=torch.device('cpu')))
encoder.eval()

decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout).to(device)
decoder.load_state_dict(torch.load(r'C:\Users\grigo\OneDrive\Desktop\nlpcv2\decoder.pth',map_location=torch.device('cpu')))
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)


nlp = spacy.load("en_core_web_sm")


chatbot = GenericAssistant('intents_two.json')
chatbot.train_model()
chatbot.save_model()
print("Bot runnins...")

client = discord.Client()
load_dotenv()
TOKEN = os.getenv('TOKEN')



@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    print('Bot is online')

flag = False

@client.event
async def on_message(message):
    global flag
    if message.author == client.user:
        return print('A fost transmis raspunsul catre utilizator')
    if message.content.startswith(""):
        if flag == True:
            message_indices = vocabulary.sentence2indices(message.content)
            print(message_indices)
            result = model(torch.tensor(message_indices).view(1,-1))
            index_sentiment = torch.argmax(result)
            index_to_sentiment = {0:'Positive',1 : 'Neutral', 2: 'Negative'}
            emotion = index_to_sentiment[index_sentiment.item()]
            positive_list = ['You should visit La Dolce Italia, Strada Mitropolit Varlaam 75, enjoy the best ice cream around you!','Try to visit Ciao Cacao, Strada Arborilor 21, you will never regret','Visit Ice Dessert, Короленко 3/2, very delicios ice cream!']
            neutral_list = ['You should visit Casa della Pizza at Bulevardul Alexandru Cel Bun 42', 'Try to visit Andys Pizza, Lev Tolstoi 24/1 street', ' Visit Torro Burgers at Strada Trandafirilor 43, after your mood should increase!']
            negative_list = ['Try to visit Brothers Pub, Strada Mihai Eminescu 29, this will normalize your negative energy','You should visit 3 Little Pigs, Strada București 67, this should decrease your sad emotions','Visit MadMans Pub, Strada Alexei Şciusev 39, positive is here!!!']
            print(emotion)


            if emotion == 'Positive':
                
                response = random.choice(positive_list)
            elif emotion == 'Neutral':
                response = random.choice(neutral_list)
            elif emotion == 'Negative':
                response = random.choice(negative_list)

            flag = False
        else:

            response = chatbot._predict_class(message.content)
            intent = response[0]['intent']
            if intent == 'other':
                response = ' '.join([elem for elem in evalueate_input(message.content, encoder, decoder, searcher, voc) if elem!='.'])
                print('other')
            elif intent == 'weather':
                print('weather')
                doc = nlp(message.content) 
                for ent in doc.ents:
                    if ent.label_ in ['GPE', 'LOC']:
            
                        url = "https://api.openweathermap.org/data/2.5/weather?q={}&appid=288af201969c4233798017e17354af37".format(ent.text)

                        res = requests.get(url)
                        data = res.json()

                        temp = data['main']['temp']
                        temp_c = (temp - 272)
                        wind = data['wind']['speed']
                        humidity = data['main']['humidity']


                        response = f"""Temperature:{int(temp_c)}
Wind Speed: {wind}
Humidity: {humidity} """
            elif intent == 'food':
                response = 'What is your mood today?'
                flag = True

                print('food')
        #response = chatbot.request(message.content[:])
        await message.channel.send(response)
client.run('OTYwNDk4MzE3NDg4NDMxMTM1.YkrTxA.UrI-WpqwZFmKv6fuqBPuHChzOyI')


nlp = spacy.load("en_core_web_sm")



