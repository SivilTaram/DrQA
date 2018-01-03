import logging

"""
Preprocess Config
"""
TRAINING_FILE = 'SQuAD/train-v1.1.json'
DEV_FILE = 'SQuAD/dev-v1.1.json'
EMBEDDING_FILE = 'embedding/glove.840B.300d.txt'
EMBEDDING_DIM = 300

UNK_ID = 1
PADDING_ID = 0

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
prep_logger = logging.getLogger(__name__)

"""
Log Config
"""
LOG_FILE = 'output.log'
# 每过10个batch打一次log记录loss
LOG_PRE_BATCH = 10

DATA_FILE = "SQuAD/data.msgpack"
META_FILE = "SQuAD/meta.msgpack"

MODEL_DIR = "Model"
MAX_CHECKPOINT = 10
# 过1个epoch评估1次性能
EVAL_PER_EPOCH = 1

"""
Train Config
"""
MAX_EPOCH = 20
BATCH_SIZE = 32
RESUME = True

USE_GPU = False
# 微调词向量前1000
TUNE_PARTIAL = 1000

"""
Predict Config
"""
TRANS_FILE = "SQuAD/transform.msgpack"

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
