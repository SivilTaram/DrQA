import os
import spacy
import json
import string
from collections import Counter
import msgpack
import math
import re
import config
import numpy as np

class Processor:
    def __init__(self):
        self.nlp = spacy.load('en', parser=False)
        # 构建停用词表
        with open(config.STOPWORD_FILE) as f:
            # 加上标点符号的过滤
            self.stopwords = set(map(lambda x: x.strip(), f.readlines()))
        self.punc = set(string.punctuation)

    def normalize_text(self, text):
        def get_lemma(token):
            if token.lemma_ != '-PRON-':
                return token.lemma_
            else:
                return token.text.lower()
        text = text.lower()
        text = ''.join(ch for ch in text if ch not in self.punc)
        text = re.sub('\s+', ' ', text)
        tokens = self.nlp(text)
        tokens = list(map(lambda t: get_lemma(t), filter(lambda token: token.text not in self.stopwords, tokens)))
        return tokens


class Indexer:
    """
    实现倒排索引
    """
    def __init__(self, folder):
        self.folder = folder
        self.processor = Processor()
        self.inverted_index = {}
        # 所有文档的数量
        self.doc_len = 0
        # 在最后归一化构建为IDF的值
        self.idf = {}

    def build_index(self):
        # 从 folder 中读取文档
        files = sorted(os.listdir(self.folder))
        for file in files:
            # 为文件构建索引
            file_path = os.path.join(self.folder, file)
            with open(file_path, encoding='utf8') as f:
                data = json.loads(f.read())
                id_ = data['id']
                all_tokens = self.processor.normalize_text(' '.join(data['paragraphs']))
                counter = Counter(all_tokens)
                # 段落总词数
                doc_words = len(counter.keys())
                for i, token in enumerate(all_tokens):
                    if token not in self.inverted_index.keys():
                        self.inverted_index[token] = []
                    self.inverted_index[token].append((id_, counter[token] / doc_words, i))
                for word in counter.keys():
                    self.idf[word] = self.idf.get(word, 0) + 1
            self.doc_len += 1
            if self.doc_len % 100 == 0:
                print("Process:{0}".format(self.doc_len))
        # 归一化IDF值
        for word in self.idf.keys():
            self.idf[word] = math.log(self.doc_len + 1 / self.idf[word])

    def dumps(self):
        dump_file = os.path.join(self.folder, 'wiki.index')
        index = {
            "IDF": self.idf,
            "Inverted": self.inverted_index
        }
        with open(dump_file, 'wb') as f:
            msgpack.dump(index, f)

    def restore(self):
        dump_file = os.path.join(self.folder, 'wiki.index')
        with open(dump_file, 'rb') as f:
            index = msgpack.load(f, encoding='utf8')
        self.inverted_index = index["Inverted"]
        self.idf = index["IDF"]

    def search(self, question):
        # 去掉停用词的问题词组
        ques_tokens = self.processor.normalize_text(question)
        # 如果没有加载索引文件，则先加载
        if len(self.idf.keys()) == 0:
            self.restore()
        ids = None
        # 文档名到词的tf映射
        tf_map = {}
        # 离散标准差
        sf_map = {}
        for token in ques_tokens:
            if token in self.inverted_index:
                index_ids = self.inverted_index[token]
            else:
                ids = {}
                break
            token_decay = {}
            # 每篇文章初始衰减系数为1
            for index_id in index_ids:
                token_decay.setdefault(index_id[0], 1.0)
            for index_id in index_ids:
                # 计算 tf-idf 值，利于后面的排序
                file_name = index_id[0]
                # 记录每个 id 中 每个 token 的 tf-idf 值
                tf_map[file_name] = tf_map.get(file_name, 0) + self.idf[token] * index_id[1] * token_decay[file_name]
                token_decay[file_name] /= 2
                # 计算文章的离散度
                if file_name not in sf_map.keys():
                    sf_map[file_name] = []
                # 加入位置信息，用于计算离散度
                sf_map[file_name].append(index_id[2])
            token_set = set(map(lambda x: x[0], index_ids))
            if ids is None:
                ids = token_set
            else:
                ids &= token_set
        # tf-idf 系数减去离散标准差
        for file_name in tf_map.keys():
            if len(sf_map[file_name]) > 2:
                tf_map[file_name] -= math.log(1 + np.std(sf_map[file_name]))
        # 获得文件ID, 进行相关性排序
        sorted_files = list(filter(lambda x: x in ids, map(lambda x: x[0], sorted(tf_map.items(), key=lambda d: d[1], reverse=True))))[:5]
        # 寻找排序文件前3的各个段落
        contexts = {}
        for file in sorted_files:
            with open(os.path.join(self.folder, file + ".txt")) as f:
                data = json.load(f)
            for i, paragraph in enumerate(data['paragraphs']):
                paragraph_tokens = set(self.processor.normalize_text(paragraph))
                votes = 0
                for token in ques_tokens:
                    if token in paragraph_tokens:
                        votes += 1
                if votes >= 0.6 * len(ques_tokens) and len(paragraph) < 1500:
                    contexts[paragraph] = votes
        # 对段落的共现进行排序
        sorted_context = list(sorted(contexts.keys(), key=lambda d: d[1], reverse=True))[:20]
        return sorted_context


if __name__ == "__main__":
    indexer = Indexer("Demo")
    indexer.build_index()
    indexer.dumps()