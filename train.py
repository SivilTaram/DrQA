import re
import os
import sys
import random
import string
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
from config import *

parser = argparse.ArgumentParser(
    description='Train a Document Reader model.'
)
# system
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=3,
                    help='log model loss per x updates (mini-batches).')
parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('--save_last_only', action='store_true',
                    help='only save the final models.')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='perform evaluation per x epochs.')
parser.add_argument('--seed', type=int, default=1706123,
                    help='random seed for data shuffling')
# training
parser.add_argument('-rs', '--resume', default='',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
# model
parser.add_argument('--doc_layers', type=int, default=3)
parser.add_argument('--question_layers', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=4)

parser.add_argument('--dropout_emb', type=float, default=0.4)

parser.add_argument('--max_len', type=int, default=15)

args = parser.parse_args()

# set model dir

os.makedirs(MODEL_DIR, exist_ok=True)
model_dir = os.path.abspath(MODEL_DIR)

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if USE_GPU:
    torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def main():
    log.info('[program starts.]')
    train, dev, dev_y, embedding, option = load_data(vars(args))
    log.info(option)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.resume))
        if args.resume_options:
            option = checkpoint['config']
        resume_dict = checkpoint['resume_dict']
        model = DocReaderModel(option, embedding, resume_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    else:
        model = DocReaderModel(option, embedding)
        epoch_0 = 1

    if USE_GPU:
        model.enable_gpu()

    if args.resume:
        batches = BatchGen(dev, evaluation=True)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        em, f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + MAX_EPOCH):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(dev, evaluation=True)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
            em, f1 = score(predictions, dev_y)
            log.warning("dev EM: {} F1: {}".format(em, f1))
        # save
        if not args.save_last_only or epoch == epoch_0 + MAX_EPOCH - 1:
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer


def load_data(option):
    # 加载各项词汇表
    with open(META_FILE, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['Embedding'])
    option['pretrained_words'] = True
    # 自动获取embedding的维度与vocabulary大小
    option['vocab_size'] = embedding.size(0)
    option['embedding_dim'] = embedding.size(1)
    option['pos_size'] = len(meta['POSTag'])
    option['ner_size'] = len(meta['NamedEntity'])
    # 加载训练集与测试集
    with open(DATA_FILE, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = data['train']
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, option


class BatchGen:
    def __init__(self, data, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = BATCH_SIZE
        self.eval = evaluation
        self.gpu = USE_GPU

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, args.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, args.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            if not self.eval:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        """ 去除文本中的标点符号
        :param text:
        :return:
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


if __name__ == '__main__':
    main()
