"""
https://github.com/jmtomczak/intro_dgm/blob/main/llms/teenygpt_example.ipynb
"""

import os
import pickle

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np

from datasets import load_dataset

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_model_summary import summary


class DataProcessor(object):
    """
    전처리 담당
    """
    def __init__(self, ):
        super().__init__()
        nlp = spacy.load("en_core_web_sm")
        nltk.download('omw-1.4')
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("stopwords")

    @staticmethod
    def preprocess_text(text):
        # Tokenize, remove punctuation and lowercase
        tokens = nltk.word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]

        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        processed_text = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

        return " ".join(processed_text)
    
    def process_batch(self, texts):
        return [self.preprocess_text(d) for d in texts]
    

class Tokenizer():
    """
    캐릭터를 정수로 바꾸고 패딩함
    """
    def __init__(self, max_length=0):
        super().__init__()

        self.max_length = max_length

        self.alphabet_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        self.alphabet = self.prepare_alphabet()
        self.decoded_alphabet = self.prepare_decoded_alphabet()


    def prepare_alphabet(self):
        # PREPARE THE ALPHABET (CHAR->INT)
        # as a dictionary
        alphabet = {}
        alphabet['pad'] = 0  # add 'pad'
        count = 1

        for letter in self.alphabet_letters:
            alphabet[letter] = count
            count += 1

        # add 'pad', 'bos', 'eos' tokens
        alphabet[' '] = count
        alphabet['cls'] = count + 1

        return alphabet

    def prepare_decoded_alphabet(self):
        # PREPARE DECODED ALPHABET (INT->CHAR)
        decoded_alphabet_ints = [i for i in range(len(self.alphabet_letters))]

        decoded_alphabet = {}
        decoded_alphabet[0] = 'pad'

        for i in decoded_alphabet_ints:
            decoded_alphabet[i+1] = self.alphabet_letters[i]

            decoded_alphabet[i+2] = ' '
        decoded_alphabet[i+3] = 'cls'

        return decoded_alphabet

    def encode(self, texts):
        N = len(texts)

        if self.max_length == 0:
            max_length = 0
            for i in range(N):
                len_i = len(texts[i])
                if len_i > max_length:
                    max_length = len_i
        else:
            max_length = self.max_length

        tokens = np.zeros((N, max_length+1))

        for i in range(N):
            len_i = len(texts[i])
            for j in range(-1, max_length):
                if j == -1:
                    tokens[i,j+1] = self.alphabet['cls']
                elif j >= len_i:
                    tokens[i,j+1] = self.alphabet['pad']
                else:
                    if texts[i][j] == 'é':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'í':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'á':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ó':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'æ':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ä':
                        tokens[i,j+1] = self.alphabet['a']
                    else:
                        tokens[i,j+1] = self.alphabet[texts[i][j]]

        return tokens

    def decode(self, tokens):
        texts = []

        for i in range(len(tokens)):
            tokens_i = tokens[i,:]
            text_i = ''
            for j in range(len(tokens_i)):
                if tokens_i[j] == 0:
                    break
                else:
                    if self.decoded_alphabet[tokens_i[j]] != 'cls':
                        text_i += self.decoded_alphabet[tokens_i[j]]
            texts.append(text_i)

        return texts
    
class Headers(Dataset):
    """A simple dataset based on headers. Source: https://huggingface.co/datasets/IlyaGusev/headline_cause"""

    def __init__(self, dataprocessor, tokenizer, mode='train', num_training_data=None, transforms=None):
        # LOAD DATA
        dataset = load_dataset("IlyaGusev/headline_cause", "en_simple")

        # PREPARE DATA
        if mode == 'train':
            train_texts = dataprocessor.process_batch(dataset['train'][:]['left_title'] + dataset['train'][:]['right_title']) # list
            if num_training_data is None:
                self.data = torch.from_numpy(tokenizer.encode(train_texts)).long()
            else:
                self.data = torch.from_numpy(tokenizer.encode(train_texts))[:num_training_data].long()
        elif mode == 'val':
            validation_texts = dataprocessor.process_batch(dataset['validation'][:]['left_title'] + dataset['validation'][:]['right_title']) # list
            self.data = torch.from_numpy(tokenizer.encode(validation_texts)).long()
        else:
            test_texts = dataprocessor.process_batch(dataset['test'][:]['left_title'] + dataset['test'][:]['right_title']) # list
            self.data = torch.from_numpy(tokenizer.encode(test_texts)).long()

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample


# Loss Function (NLL) for the categorical distribution
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss(reduction='none')

    def forward(self, y_model, y_true, reduction='sum'):
        # y_model: B(atch) x T(okens) x V(alues)
        # y_true: B x T
        B, T, V = y_model.size()

        y_model = y_model.view(B * T, V)
        y_true = y_true.view(B * T,)

        loss_matrix = self.loss(y_model, y_true) # B*T

        if reduction == 'sum':
            return torch.sum(loss_matrix)
        elif reduction == 'mean':
            loss_matrix = loss_matrix.view(B, T)
            return torch.mean(torch.sum(loss_matrix, 1))
        else:
            raise ValueError('Reduction could be either `sum` or `mean`.')


class TransformerBlock(nn.Module):
    def __init__(self, num_emb, num_neurons, num_heads=4):
        super().__init__()

        # hyperparams
        self.D = num_emb
        self.H = num_heads
        self.neurons = num_neurons

        # components
        self.msha = nn.MultiheadAttention(embed_dim=self.D, num_heads=self.H, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(self.D)
        self.layer_norm2 = nn.LayerNorm(self.D)

        self.mlp = nn.Sequential(nn.Linear(self.D, self.neurons * self.D),
                                nn.GELU(),
                                nn.Linear(self.neurons * self.D, self.D))

    def forward(self, x, causal=True):
        # Multi-Head Self-Attention
        x_attn, _ = self.msha(x, x, x, is_causal=causal, attn_mask=torch.empty(1,1), need_weights=False)
        # LayerNorm
        x = self.layer_norm1(x_attn + x)
        # MLP
        x_mlp = self.mlp(x)
        # LayerNorm
        x = self.layer_norm2(x_mlp + x)

        return x
    

class teenyGPT(nn.Module):
    def __init__(self, num_tokens, num_token_vals, num_emb, num_neurons, num_heads=2, dropout_prob=0.1, num_blocks=10, device='cpu'):
        super().__init__()

        # Remember, always credit the author, even if it's you ;)
        print('teenyGPT by JT.')

        # hyperparams
        self.device = device
        self.num_tokens = num_tokens
        self.num_token_vals = num_token_vals
        self.num_emb = num_emb
        self.num_blocks = num_blocks

        # embedding layer
        self.embedding = torch.nn.Embedding(num_token_vals, num_emb)

        # positional embedding
        self.positional_embedding = nn.Embedding(num_tokens, num_emb)

        # transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(num_emb=num_emb, num_neurons=num_neurons, num_heads=num_heads))

        # output layer (logits + softmax)
        self.logits = nn.Sequential(nn.Linear(num_emb, num_token_vals))

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # loss function
        self.loss_fun = LossFun()

    def transformer_forward(self, x, causal=True, temperature=1.0):
        # x: B(atch) x T(okens)
        # embedding of tokens
        x = self.embedding(x) # B x T x D
        # embedding of positions
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0).to(self.device)
        pos_emb = self.positional_embedding(pos)
        # dropout of embedding of inputs
        x = self.dropout(x + pos_emb)

        # transformer blocks
        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x)

        # output logits
        out = self.logits(x)

        return F.log_softmax(out/temperature, 2)

    @torch.no_grad()
    def sample(self, batch_size=4, temperature=1.0):
        x_seq = np.asarray([[self.num_token_vals - 1] for i in range(batch_size)])

        # sample next tokens
        for i in range(self.num_tokens-1):
            xx = torch.tensor(x_seq, dtype=torch.long, device=self.device)
            # process x and calculate log_softmax
            x_log_probs = self.transformer_forward(xx, temperature=temperature)
            # sample i-th tokens
            x_i_sample = torch.multinomial(torch.exp(x_log_probs[:,i]), 1).to(self.device)
            # update the batch with new samples
            x_seq = np.concatenate((x_seq, x_i_sample.to('cpu').detach().numpy()), 1)

        return x_seq

    @torch.no_grad()
    def top1_rec(self, x, causal=True):
        x_prob = torch.exp(self.transformer_forward(x, causal=True))[:,:-1,:].contiguous()
        _, x_rec_max = torch.max(x_prob, dim=2)
        return torch.sum(torch.mean((x_rec_max.float() == x[:,1:].float().to(device)).float(), 1).float())

    def forward(self, x, causal=True, temperature=1.0, reduction='mean'):
        # get log-probabilities
        log_prob = self.transformer_forward(x, causal=causal, temperature=temperature)

        return self.loss_fun(log_prob[:,:-1].contiguous(), x[:,1:].contiguous(), reduction=reduction)
    
    
if __name__ == '__main__':

    dataset = load_dataset("IlyaGusev/headline_cause", "en_simple")


    dataprocessor = DataProcessor()
    tokenizer = Tokenizer(max_length=149)

    #-dataset
    num_training_data = None
    train_dataset = Headers(dataprocessor, tokenizer, num_training_data=num_training_data, mode="train")
    validation_dataset = Headers(dataprocessor, tokenizer, mode="val")
    test_dataset = Headers(dataprocessor, tokenizer, mode="test")