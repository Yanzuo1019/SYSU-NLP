# Author: Lu Yanzuo
# Date: 2020-11-26
# Description: CWS BiLSTM-CRF Model

import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def log_sum_exp(score):
    max_score = score[0, torch.max(score, 1)[1].item()]
    max_score_expand = max_score.view(1, -1).expand(1, score.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(score - max_score_expand)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_size, hidden_dim, device, embedding_list):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.target_size = len(self.tag2id)

        self.word_embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_dim // 2, bidirectional=True)

        self.emission = nn.Linear(hidden_dim, self.target_size)
        self.transition = nn.Parameter(torch.randn(self.target_size, self.target_size))
        self.transition.data[tag2id[START_TAG], :] = -10000.0
        self.transition.data[:, tag2id[STOP_TAG]] = -10000.0

        if embedding_list is not None:
            self.word_embed.from_pretrained(torch.Tensor(embedding_list))
            # for para in self.word_embed.parameters():
            #     para.requires_grad = False

    def init_hidden(self):
        hidden = (torch.randn(2, 1, self.hidden_dim // 2).to(self.device), \
            torch.randn(2, 1, self.hidden_dim // 2).to(self.device))
        return hidden
    
    def forward(self, sentence, tags):
        hidden = self.init_hidden()
        embedding = self.word_embed(sentence).view(len(sentence), 1 , -1)
        lstm_feat, _ = self.lstm(embedding, hidden)
        lstm_feat = lstm_feat.view(len(sentence), self.hidden_dim)
        feats = self.emission(lstm_feat)
        
        forward_var = torch.full((1, self.target_size), -10000.0).to(self.device)
        forward_var[0][self.tag2id[START_TAG]] = 0.0
        for feat in feats:
            alpha_t = []
            for next_tag in range(self.target_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
                trans_score = self.transition[next_tag].view(1, -1)
                score = forward_var + emit_score + trans_score
                alpha_t.append(log_sum_exp(score).view(1))
            forward_var = torch.cat(alpha_t).view(1, -1)
        score = forward_var + self.transition[self.tag2id[STOP_TAG]]
        log_sum_exp_score = log_sum_exp(score)

        gt_score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.LongTensor([self.tag2id[START_TAG]]).to(self.device), tags])
        for i, feat in enumerate(feats):
            gt_score += self.transition[tags[i+1], tags[i]] + feat[tags[i+1]]
        gt_score += self.transition[self.tag2id[STOP_TAG], tags[-1]]
        return log_sum_exp_score, gt_score
    
    def decode(self, sentence):
        hidden = self.init_hidden()
        embedding = self.word_embed(sentence).view(len(sentence), 1 , -1)
        lstm_feat, _ = self.lstm(embedding, hidden)
        lstm_feat = lstm_feat.view(len(sentence), self.hidden_dim)
        feats = self.emission(lstm_feat)

        path_saving = []

        forward_var = torch.full((1, self.target_size), -10000.0).to(self.device)
        forward_var[0][self.tag2id[START_TAG]] = 0.0
        for feat in feats:
            max_score_id_list = []
            max_score_list = []
            for next_tag in range(self.target_size):
                score = forward_var + self.transition[next_tag]
                max_score_id = torch.max(score, 1)[1].item()
                max_score_id_list.append(max_score_id)
                max_score_list.append(score[0][max_score_id].view(1))
            forward_var = (torch.cat(max_score_list) + feat).view(1, -1)
            path_saving.append(max_score_id_list)
        score = forward_var + self.transition[self.tag2id[STOP_TAG]]
        max_score_id = torch.max(score, 1)[1].item()
        max_score = score[0][max_score_id]

        best_path = [max_score_id]
        for path in reversed(path_saving):
            max_score_id = path[max_score_id]
            best_path.append(max_score_id)
        best_path = best_path[:-1]
        best_path.reverse()

        return best_path