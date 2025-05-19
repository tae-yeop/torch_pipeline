import os
import requests
import numpy as np

import tiktoken
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from datasets import load_dataset

from pydantic import BaseModel
from typing import Literal

class DataProcessorConfig(BaseModel):
    dataset: Literal['omw-1.4', 'headline_cause']

class DataProcessor(object):
    def __init__(self, cfg: DataProcessorConfig):
        self.cfg = cfg
        if self.cfg.dataset == 'shakespeare':
            self.process_shakespeare()
        elif self.cfg.dataset == 'headline_cause':
            self.process_headlines()
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")

    def process_shakespeare(self):
        input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()

        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode with tiktoken gpt2 bpe
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
        val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

        # train.bin has 301,966 tokens
        # val.bin has 36,059 tokens 

    def process_headlines(self):
        # 영어 토크나이저·품사·lemmatizer 등이 들어 있는 사전 학습 파이프라인
        # 향후 품사태깅·의존구문 분석 등을 추가하려고 미리 로드
        nlp = spacy.load("en_core_web_sm")
        # NLTK WordNetLemmatizer 가 어휘 변형 규칙을 참조할 때 필요.
        nltk.download('omw-1.4')
        # NLTK의 문장·단어 토크나이저 모델. nltk.word_tokenize() 가 내부적으로 사용
        nltk.download("punkt")
        # 영어 WordNet 사전 : lemmatizer가 “run→run, running→run” 처럼 원형 복원할 때 참조
        nltk.download("wordnet")
        # 불용어 리스트(“the, and, of …”). stopwords.words("english") 호출 시 필요.
        nltk.download("stopwords")

        lemmatizer = WordNetLemmatizer()
        # Download the dataset
        dataset = load_dataset("IlyaGusev/headline_cause", "en_simple")

        data_to_be_processed = dataset['train'][:]['left_title'] + dataset['train'][:]['right_title']

        data_to_be_processed = [str(i) for i in data_to_be_processed]
        data_processed = []
        for data in data_to_be_processed:
            # Tokenize, remove punctuation and lowercase
            tokens = nltk.word_tokenize(data)
            tokens = [token for token in tokens if token.isalpha()]

            # Remove stopwords and lemmatize
            stop_words = set(stopwords.words("english"))

            processed_text = [
                lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
            ]

            data_processed.append(processed_text)