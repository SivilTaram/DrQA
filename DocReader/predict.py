from DocReader.model.batch_helper import BatchGen
import config
from DocReader.preprocess import annotate
import msgpack
import os
import torch
from DocReader.model.model import DocReaderModel


class Predictor:
    def __init__(self):
        self.gpu = config.USE_GPU
        # 加载encoders
        with open(config.TRANS_FILE, 'rb') as f:
            encoders = msgpack.load(f, encoding='utf8')
        self.vocab_encoder = encoders['VocabEncoder']
        self.pos_encoder = encoders['POSEncoder']
        self.entity_encoder = encoders['EntityEncoder']
        # 加载embedding
        with open(config.META_FILE, 'rb') as f:
            meta = msgpack.load(f, encoding='utf8')
        embedding = torch.Tensor(meta['Embedding'])

        # 初始化模型
        option = {'vocab_size': embedding.size(0), 'embedding_dim': embedding.size(1)}
        model_file = os.path.join(config.MODEL_DIR, config.PREDICT_MODEL)

        # 加载模型，注意 GPU 张量的改变
        checkpoint = torch.load(model_file,  map_location=lambda storage, loc: storage)
        resume_dict = checkpoint['resume_dict']
        self.model = DocReaderModel(option, embedding, resume_dict)

    def get_prediction(self, question, context):
        # 构造row
        row = (1, context, question)
        features = annotate(row)
        assert len(features) == 8
        # 构造因子化后的feature
        question_tokens = features[5]
        context_tokens = features[1]
        context_features = features[2]
        context_tags = features[3]
        context_ents = features[4]
        question_ids = [self.vocab_encoder[w] if w in self.vocab_encoder else config.UNK_ID for w in question_tokens]
        context_ids = [self.vocab_encoder[w] if w in self.vocab_encoder else config.UNK_ID for w in context_tokens]
        pos_dummy = [self.pos_encoder[w] for w in context_tags]
        entity_dummy = [self.entity_encoder[w] for w in context_ents]
        factorized_feature = (
            features[0],
            context_ids,
            context_features,
            pos_dummy,
            entity_dummy,
            question_ids
        ) + features[6:]
        # 将因子化后的数据转换成一个batch
        predictions = []
        batches = BatchGen([factorized_feature], evaluation=True)
        for batch in batches:
            predictions.extend(self.model.predict(batch))
        return predictions[0]


if __name__ == "__main__":
    predictor = Predictor()
    while True:
        context = input("Please input context:\n")
        question = input("Please input question:\n")
        res = predictor.get_prediction(question, context)
        print("Predict: {0}".format(res))