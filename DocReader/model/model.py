# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from .utils import AverageMeter
from .rnn_reader import RnnDocReader
import config

logger = config.logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, resume_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = resume_dict['updates'] if resume_dict else 0
        self.train_loss = AverageMeter()

        # Building network.
        self.network = RnnDocReader(opt, embedding=embedding)
        if resume_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(resume_dict['network'].keys()):
                if k not in new_state:
                    del resume_dict['network'][k]
            self.network.load_state_dict(resume_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(parameters,
                                      weight_decay=config.WEIGHT_DECAY)

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if config.USE_GPU:
            inputs = [Variable(e.enable_gpu(async=True)) for e in ex[:7]]
            target_s = Variable(ex[7].enable_gpu(async=True))
            target_e = Variable(ex[8].enable_gpu(async=True))
        else:
            inputs = [Variable(e) for e in ex[:7]]
            target_s = Variable(ex[7])
            target_e = Variable(ex[8])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(), config.GRAD_CLIPPING)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if config.USE_GPU:
            inputs = [Variable(e.enable_gpu(async=True), volatile=True)
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        max_len = config.MAX_SPAN_LEN or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if config.TUNE_PARTIAL > 0:
            offset = config.TUNE_PARTIAL + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'resume_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('Save model to {0}'.format(filename))
        except Exception as e:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')
            logger.warning('[ StackTrace: {0}]'.format(e))

    def enable_gpu(self):
        self.network.cuda()
