import logging


TRAINING_FILE = 'SQuAD/train-v1.1.json'
DEV_FILE = 'SQuAD/dev-v1.1.json'
EMBEDDING_FILE = 'embedding/glove.840B.300d.txt'
EMBEDDING_DIM = 300

UNK_ID = 1
PADDING_ID = 0

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
logger = logging.getLogger(__name__)
