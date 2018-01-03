# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers
from config import *


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0),
                                          embedding.size(1),
                                          padding_idx=padding_idx)
            self.embedding.weight.data[2:, :] = embedding[2:, :]
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
        # Projection for attention weighted question
        self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        doc_input_size += opt['embedding_dim']
        doc_input_size += opt['pos_size']
        doc_input_size += opt['ner_size']

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=DOC_LAYER,
            dropout_rate=RNN_DROPOUT_RATE,
            dropout_output=OUTPUT_DROPOUT,
            concat_layers=CONTACT_RNN_LAYER,
            padding=RNN_PADDING,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],
            hidden_size=HIDDEN_SIZE,
            num_layers=QUES_LAYER,
            dropout_rate=RNN_DROPOUT_RATE,
            dropout_output=OUTPUT_DROPOUT,
            concat_layers=CONTACT_RNN_LAYER,
            padding=RNN_PADDING,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * HIDDEN_SIZE
        question_hidden_size = 2 * HIDDEN_SIZE
        if CONTACT_RNN_LAYER:
            doc_hidden_size *= DOC_LAYER
            question_hidden_size *= QUES_LAYER

        # Question merging
        self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if EMBEDDING_DROPOUT_RATE > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=EMBEDDING_DROPOUT_RATE,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=EMBEDDING_DROPOUT_RATE,
                                           training=self.training)

        drnn_input_list = [x1_emb, x1_f]
        # 增加带注意力的问题表示
        x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
        drnn_input_list.append(x2_weighted_emb)
        # 增加 POS Feature
        drnn_input_list.append(x1_pos)
        # 增加 NER Feature
        drnn_input_list.append(x1_ner)
        drnn_input = torch.cat(drnn_input_list, 2)
        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
