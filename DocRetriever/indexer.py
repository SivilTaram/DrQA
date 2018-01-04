import os
import spacy
import json
import string
from collections import Counter
import msgpack
import math
import re
import config


class Processor:
    def __init__(self):
        self.nlp = spacy.load('en', parser=False)
        # 构建停用词表
        with open(config.STOPWORD_FILE) as f:
            # 加上标点符号的过滤
            self.stopwords = set(map(lambda x: x.strip(), f.readlines()))
        self.punc = set(string.punctuation)

    def normalize_text(self, text):
        text = text.lower()
        text = ''.join(ch for ch in text if ch not in self.punc)
        text = re.sub('\s+', ' ', text)
        tokens = self.nlp(text)
        tokens = list(map(lambda t: t.text, filter(lambda token: token.text not in self.stopwords, tokens)))
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
                for token in counter.keys():
                    if token not in self.inverted_index.keys():
                        self.inverted_index[token] = []
                    self.inverted_index[token].append((id_, counter[token] / doc_words))
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
        for token in ques_tokens:
            index_ids = self.inverted_index[token]
            for index_id in index_ids:
                # 计算 tf-idf 值，利于后面的排序
                file_name = index_id[0]
                # 记录每个 id 中 每个 token 的 tf-idf 值
                tf_map[file_name] = tf_map.get(file_name, 0) + self.idf[token] * index_id[1]
            token_set = set(map(lambda x: x[0], index_ids))
            if ids is None:
                ids = token_set
            else:
                ids &= token_set
        # 获得文件ID, 进行相关性排序, 取前三个
        sorted_files = list(filter(lambda x: x in ids, sorted(tf_map.keys(), key=lambda d: d[1], reverse=True)))[:3]
        # 寻找排序文件前3的各个段落
        contexts = []
        for file in sorted_files:
            with open(os.path.join(self.folder, file + ".txt")) as f:
                data = json.load(f)
            for paragraph in data['paragraphs']:
                for token in ques_tokens:
                    if token in paragraph:
                        contexts.append(paragraph)
                        break
        return contexts
