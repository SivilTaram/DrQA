import re
import json
import spacy
from config import *
import msgpack
import collections
import unicodedata
import numpy as np

nlp = spacy.load('en', parser=False)


class PackData:
    def __init__(self):
        self.train_data = None
        self.dev_data = None
        # 单词 词汇表
        self.vocab = None
        # Token -> Integer 映射表
        self.vocab_encoder = None
        # POS Tags 词汇表
        self.vocab_pos = list(nlp.tagger.tag_names)
        # Tags -> Integer 映射表
        self.pos_encoder = {pos: i for i, pos in enumerate(self.vocab_pos)}
        # Entity Tags 词汇表
        self.vocab_entity = None
        # Entity -> Integer 映射表
        self.entity_encoder = None
        # 词向量映射表
        self.vector_vocab = None
        #
        self.embedding = None

    def clean_normalize_data(self):
        logger.info('Start Data Prepocessing...')
        train_data = file_formatter(TRAINING_FILE)
        dev_data = file_formatter(DEV_FILE)
        logger.info('File Convert Success.')
        logger.info('Train Sample Data:{0}'.format(train_data[0]))
        logger.info('Dev Sample Data:{0}'.format(dev_data[0]))

        # 训练集与验证集做标注
        train_data = list(map(annotate, train_data))
        self.dev_data = list(map(annotate, dev_data))
        # 过滤掉与分词结果边界划分不一致的句子
        self.train_data = filter_inconsistent(train_data)
        logger.info('Clean Data and Normalized.')

    def build_vector_vocab(self):
        # 加载词向量文件
        vector_vocab = set()
        with open(EMBEDDING_FILE, encoding='utf8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token = normalize_text(elems[0])
                vector_vocab.add(token)
        self.vector_vocab = vector_vocab

    def build_embeddings(self):
        # 针对训练而言，压缩词向量体积；如果是测试，仍然要使用全部词向量
        vocab_size = len(self.vocab)
        embeddings = np.zeros((vocab_size, EMBEDDING_DIM))
        embed_counts = np.zeros(vocab_size)
        # 指定编码，否则默认为 gbk 编码格式
        with open(EMBEDDING_FILE, encoding='utf8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token = normalize_text(elems[0])
                if token in self.vocab_encoder:
                    # 词向量在训练时是可能会出现未做词性还原等情形的
                    token_id = self.vocab_encoder[token]
                    embed_counts[token_id] += 1
                    embeddings[token_id] += [float(v) for v in elems[1:]]
        self.embedding = embeddings / embed_counts.reshape((-1, 1))
        logger.info('Loaded Embedding File Successfully.')

    def build_vocab(self):
        full_data = self.full_data
        # row[5] 是 question_tokens, row[1] 是context_tokens
        questions = [row[5] for row in full_data]
        contexts = [row[1] for row in full_data]
        question_counter = collections.Counter([token
                                                for question in questions
                                                for token in question])
        context_counter = collections.Counter([token
                                               for context in contexts
                                               for token in context])
        counter = context_counter + question_counter
        vocab = sorted([t for t in question_counter if t in self.vector_vocab], key=question_counter.get,
                       reverse=True)
        vocab += sorted([t for t in context_counter.keys() - question_counter.keys() if t in self.vector_vocab],
                        key=counter.get, reverse=True)
        vocab.insert(PADDING_ID, '<PAD>')
        vocab.insert(UNK_ID, '<UNK>')
        self.vocab = vocab
        self.vocab_encoder = {w: i for i, w in enumerate(self.vocab)}

    def init_transforms(self):
        full_data = self.full_data
        entity_counter = collections.Counter(w for row in full_data for w in row[4])
        self.vocab_entity = sorted(entity_counter, key=entity_counter.get, reverse=True)
        logger.info('Vocabulary size: {0}'.format(len(self.vocab)))
        logger.info('Found {0} Entity tags: {1}'.format(len(self.vocab_entity), self.vocab_entity))

        self.pos_encoder = {w: i for i, w in enumerate(self.vocab_pos)}
        self.entity_encoder = {w: i for i, w in enumerate(self.vocab_entity)}

    def factorize(self, row):
        question_tokens = row[5]
        context_tokens = row[1]
        context_features = row[2]
        context_tags = row[3]
        context_ents = row[4]
        question_ids = [self.vocab_encoder[w] if w in self.vocab_encoder else UNK_ID for w in question_tokens]
        context_ids = [self.vocab_encoder[w] if w in self.vocab_encoder else UNK_ID for w in context_tokens]
        pos_dummy = [self.pos_encoder[w] for w in context_tags]
        entity_dummy = [self.entity_encoder[w] for w in context_ents]
        return (row[0], context_ids, context_features, pos_dummy, entity_dummy, question_ids) + row[6:]

    def transform(self):
        self.train_data = list(map(self.factorize, self.train_data))
        self.dev_data = list(map(self.factorize, self.dev_data))
        logger.info('POS, Entity, Vocabulary Factorized & Transform.')

    def dumps(self):
        meta = {
            'Vocab': self.vocab,
            'POSTag': self.vocab_pos,
            'NamedEntity': self.vocab_entity,
            'Embedding': self.embedding.tolist()
        }
        with open('SQuAD/meta.msgpack', 'wb') as f:
            msgpack.dump(meta, f)

        result = {
            'train': self.train_data,
            'dev': self.dev_data
        }
        with open('SQuAD/data.msgpack', 'wb') as f:
            msgpack.dump(result, f)

        trans = {
            'VocabEncoder': self.vocab_encoder,
            'POSEncoder': self.pos_encoder,
            'EntityEncoder': self.entity_encoder
        }

        with open('SQuAD/transform.msgpack', 'wb') as f:
            msgpack.dump(trans, f)

        logger.info('Save Data to Disk.')

    @property
    def full_data(self):
        return self.train_data + self.dev_data


def file_formatter(file_name):
    """ 将训练集和测试集的格式标准化。训练集中 answer 只会有一个，测试集中 answer 可能有多个。

    INPUT:
    {
        "data": [
            {
                "title": "Super_Bowl_50",
                "paragraphs": [
                    {
                        "context": "Super Bowl 50 was an American football game to determine the champion of the
                        National Football League (NFL) for the ... so that the logo could prominently  feature the
                        Arabic numerals 50.",
                        "qas": [
                            {
                                "answers": [
                                    {
                                        "answer_start": 177,
                                        "text": "Denver Broncos"
                                    },
                                    ...
                                    {
                                        "answer_start": 177,
                                        "text": "Denver Broncos"
                                    }
                                ],
                                "question": "Which NFL team represented the AFC at Super Bowl 50?",
                                "id": "56be4db0acb8001400a502ec"
                            },
                            ...
                        ]
                    }
                    ...
                ]
            }
            ...
        ]
    }

    OUTPUT:
    # 训练集:
    (
        (   "56be4db0acb8001400a502ec",
            "Super Bowl 50 was ... feature the Arabic numerals 50. ",
            "Which NFL team represented the AFC at Super Bowl 50?",
            "Denver Broncos",
            177,
            191
        )
        ...
    )
    # 测试集:
    (
        (   "56be4db0acb8001400a502ec",
            "Super Bowl 50 was ... feature the Arabic numerals 50. ",
            "Which NFL team represented the AFC at Super Bowl 50?",
            ( "Denver Broncos","Denver Broncos","Denver Broncos" )
        )
        ...
    )
    :param file_name: train-v1.1.json or dev-v1.1.json
    :return: 二维表，每一行都是一个标准化后的数据
    """
    with open(file_name) as data_f:
        data_set = json.loads(data_f.read())
    qa_list = []
    for article in data_set['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if 'train' in file_name:
                    answer = answers[0]
                    answer_text = answer['text']
                    start = answer['answer_start']
                    end = start + len(answer_text)
                    qa_list.append((id_, context, question, answer_text, start, end))
                else:
                    answers = [a['text'] for a in answers]
                    qa_list.append((id_, context, question, answers))
    return qa_list


def clean_spaces(text):
    """ 将制表符等符号转换为空格 """
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def annotate(row):
    """ 对训练集和测试集文本进行feature构造
    :param row: 文本集合中的一行
    :return:
    """

    def get_lemma(token):
        if token.lemma_ != '-PRON-':
            return token.lemma_
        else:
            return token.text.lower()

    # 取出 context 与 question
    id_, context, question = row[:3]
    # 将多余的空格替换
    question_doc = nlp(clean_spaces(question))
    context_doc = nlp(clean_spaces(context))
    # 将字符按unicode重新编码
    question_tokens = [normalize_text(token.text) for token in question_doc]
    context_tokens = [normalize_text(token.text) for token in context_doc]
    # 词性标注
    context_pos = [w.tag_ for w in context_doc]
    # 命名实体识别
    context_entity = [w.ent_type_ for w in context_doc]
    # 构造 Feature
    question_tokens_lower = list(map(lambda x: x.lower(), question_tokens))
    context_tokens_lower = list(map(lambda x: x.lower(), context_tokens))
    question_lemma = set(get_lemma(token) for token in question_doc)
    # Exact Math Feature: [Origin, Lower, Lemma]
    feature_origin = list(token in question_tokens for token in context_tokens)
    feature_lower = list(token in question_tokens_lower for token in context_tokens_lower)
    feature_lemma = list(get_lemma(token) in question_lemma for token in context_doc)
    # Term Frequency Feature
    counter = collections.Counter(context_tokens_lower)
    context_len = len(context_tokens_lower)
    feature_term = [counter[token] / context_len for token in context_tokens_lower]
    context_features = list(zip(feature_origin, feature_lower, feature_lemma, feature_term))
    # 用于过滤与分词结果出现分歧的句子
    context_token_span = list(map(lambda x: (x.idx, x.idx + len(x.text)), context_doc))

    return (id_, context_tokens, context_features,
            context_pos, context_entity,
            question_tokens, context, context_token_span) + row[3:]


def filter_inconsistent(train_in):
    train = []
    for row in train_in:
        context_token_span = row[-4]
        # 寻找answer在context中出现的位置，如果与分词结果产生冲突，则舍弃
        starts, ends = zip(*context_token_span)
        answer_start = row[-2]
        answer_end = row[-1]
        try:
            train.append(row[:-3] + (starts.index(answer_start), ends.index(answer_end)))
        except ValueError:
            pass
    return train


def main():
    # 用于标准化的数据结构
    pack_data = PackData()
    # 训练集与验证集的标准化
    pack_data.clean_normalize_data()
    # 加载词向量文件
    pack_data.build_vector_vocab()
    # 初始化词汇表
    pack_data.build_vocab()
    # 初始化词向量
    pack_data.build_embeddings()
    # 初始化映射表
    pack_data.init_transforms()
    # 将所有的标签与单词向量化
    pack_data.transform()
    # 打包存储至文件
    pack_data.dumps()


if __name__ == "__main__":
    main()
