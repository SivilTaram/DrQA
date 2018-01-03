import logging
"""
Preprocess Result
"""
POS_DIM = 56
NER_DIM = 19
MANUAL_FEATURE_NUM = 4

"""
Preprocess Config
"""
TRAINING_FILE = 'DocReader/SQuAD/train-v1.1.json'
DEV_FILE = 'DocReader/SQuAD/dev-v1.1.json'
EMBEDDING_FILE = 'DocReader/embedding/glove.840B.300d.txt'
EMBEDDING_DIM = 300

UNK_ID = 1
PADDING_ID = 0

"""
Train Config
"""
MAX_EPOCH = 20
BATCH_SIZE = 32

RESUME = False
LR_DECAY = 0.1

USE_GPU = False

"""
Predict Config
"""
TRANS_FILE = "DocReader/SQuAD/transform.msgpack"
PREDICT_MODEL = "best_model.pt"

"""
Network Config
"""
RNN_PADDING = True
CONTACT_RNN_LAYER = True
OUTPUT_DROPOUT = True

EMBEDDING_DROPOUT_RATE = 0.4
RNN_DROPOUT_RATE = 0.4

DOC_LAYER = 3
QUES_LAYER = 3
HIDDEN_SIZE = 128
# 以短语作为答案的最长长度
MAX_SPAN_LEN = 15

WEIGHT_DECAY = 0
GRAD_CLIPPING = 10
# 微调词向量前1000
TUNE_PARTIAL = 1000
PRETRAIN_EMBEDDING = True

"""
Log Config
"""
TRAIN_LOG = 'train.log'
# 每过10个batch打一次log记录loss
LOG_PRE_BATCH = 10

DATA_FILE = "DocReader/SQuAD/data.msgpack"
META_FILE = "DocReader/SQuAD/meta.msgpack"

MODEL_DIR = "DocReader/checkpoints"
# 最多有多少个epoch
MAX_CHECKPOINT = 5
# 过1个epoch评估1次性能
EVAL_PER_EPOCH = 1


"""
Logger Config
"""
# proprocess logger
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
prep_logger = logging.getLogger(__name__)

# train logger
train_logger = logging.getLogger(__name__)
train_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(TRAIN_LOG)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
file_handler.setFormatter(formatter)
train_logger.addHandler(file_handler)
