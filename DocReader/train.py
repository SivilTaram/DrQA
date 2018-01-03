import os
from datetime import datetime
from shutil import copyfile
import msgpack
from DocReader.model.batch_helper import *

import config
from DocReader.model.model import DocReaderModel


def load_data():
    option = {}
    # 加载各项词汇表
    with open(config.META_FILE, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['Embedding'])
    # 自动获取embedding的维度与vocabulary大小
    option['vocab_size'] = embedding.size(0)
    option['embedding_dim'] = embedding.size(1)
    # 加载训练集与测试集
    with open(config.DATA_FILE, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = data['train']
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, option


class TrainManager:

    def __init__(self, train, dev, dev_truth, embedding):
        self.epoch_index = 1
        self.model = None
        # 如果没有模型文件夹，则新建一个
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        self.model_dir = os.path.abspath(config.MODEL_DIR)
        self.train = train
        self.dev = dev
        self.dev_truth = dev_truth
        self.embedding = embedding
        # 记录最好的f1值
        self.best_f1 = 0

    def load_model(self, option):
        config.train_logger.info('[Data Loaded.]')
        # 重建之前的模型
        if config.RESUME:
            resume_file = os.path.join(self.model_dir,'Best_Model.pt')
            if os.path.isdir(self.model_dir):
                if not os.path.isfile(resume_file):
                    # 寻找最后创建的模型，自动从该模型开始继续训练
                    files = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir)]
                    files.sort(key=lambda f: os.stat(f).st_mtime)
                    resume_file = files[-1]
            else:
                raise FileNotFoundError("Can't find the lastest checkpoint file!")

            # 降低 optimizer 的学习速率
            def lr_decay(optimizer, lr_decay):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                config.train_logger.info('[learning rate reduced by {}]'.format(lr_decay))
                return optimizer

            config.train_logger.info('[loading previous model...]')
            checkpoint = torch.load(os.path.join(self.model_dir, resume_file))
            option = checkpoint['config']
            resume_dict = checkpoint['resume_dict']
            self.model = DocReaderModel(option, self.embedding, resume_dict)
            epoch_init = checkpoint['epoch'] + 1
            self.epoch_index += epoch_init
            for i in range(checkpoint['epoch']):
                random.shuffle(list(range(len(self.train))))  # synchronize random seed
            lr_decay(self.model.optimizer, lr_decay=config.LR_DECAY)
        # 建立一个新的模型
        else:
            self.model = DocReaderModel(option, self.embedding)
        if config.USE_GPU:
            self.model.enable_gpu()

    def train_epoch(self):
        # 获取batch训练语料
        batches = BatchGen(self.train)
        start = datetime.now()
        for i, batch in enumerate(batches):
            self.model.update(batch)
            if i % config.LOG_PRE_BATCH == 0:
                config.train_logger.info('Epoch [{0:2}] Batch [{1:6}] Train Loss [{2:.5f}] Remaining[{3}]'.format(
                    self.epoch_index, self.model.updates, self.model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        self.epoch_index += 1

    def evaluate_on_dev(self):
        batches = BatchGen(self.dev, evaluation=True)
        predictions = []
        for batch in batches:
            predictions.extend(self.model.predict(batch))
        em, f1 = score(predictions, self.dev_truth)
        return em, f1

    def start_train(self):
        config.train_logger.info('[Start Training.]')
        for epoch in range(self.epoch_index, self.epoch_index + config.MAX_EPOCH):
            config.train_logger.warning('Epoch {}'.format(epoch))
            # 训练模型
            self.train_epoch()
            # 评估模型在dev集上的性能
            if epoch % config.EVAL_PER_EPOCH == 0:
                em, f1 = self.evaluate_on_dev()
                config.train_logger.warning("dev EM: {} F1: {}".format(em, f1))
            # 删除旧模型
            if epoch >= config.MAX_CHECKPOINT:
                try:
                    remove_file = os.path.join(self.model_dir,
                                               'checkpoint_epoch_{}.pt'.format(epoch - config.MAX_CHECKPOINT))
                    os.remove(remove_file)
                except Exception as e:
                    config.train_logger.error("[Delete Failed.] {0}".format(e))
            # 如果还没有达到最高epoch限制
            if not epoch == self.epoch_index + config.MAX_EPOCH - 1:
                model_file = os.path.join(self.model_dir, 'checkpoint_{}.pt'.format(epoch))
                self.model.save(model_file, epoch)
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    copyfile(model_file, os.path.join(self.model_dir, 'Best_Model.pt'))
                    config.train_logger.info('[New Best model Saved.] Epoch {0}'.format(epoch))
            else:
                config.train_logger.info('[Up to limit. Goodbye]')
                break


if __name__ == '__main__':
    # 设定随机种子，主要是防止数据每次训练时都从不同的方向进行shuffle,不利于调参
    seed = 1706123
    random.seed(seed)
    torch.manual_seed(seed)
    if config.USE_GPU:
        torch.cuda.manual_seed(seed)
    # 加载数据
    train, dev, dev_truth, embedding, opt = load_data()
    config.train_logger.info('[Data loaded.]')
    # 训练
    config.train_logger.info('[Network Config] {0}'.format(opt))
    manager = TrainManager(train=train, dev=dev, dev_truth=dev_truth, embedding=embedding)
    manager.load_model(opt)
    manager.start_train()