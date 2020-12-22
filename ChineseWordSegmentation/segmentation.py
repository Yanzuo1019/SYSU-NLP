import os
import random

import torch

from model import BiLSTM_CRF

id2tag = ['B', 'M', 'E', 'S']
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
id2word = []
word2id = {}

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag2id[START_TAG] = len(tag2id)
tag2id[STOP_TAG] = len(tag2id)


def load_data(data, x, word_num, embedding_list, embedding_size):
    for line in data:
        line = line.strip()
        if not line: continue

        line_x = []
        for i in range(len(line)):
            if line[i] == ' ': continue

            if (line[i] in id2word):
                line_x.append(word2id[line[i]])
            else:
                id2word.append(line[i])
                word2id[line[i]] = word_num
                line_x.append(word_num)
                if embedding_list is not None:
                    embedding_list.append([random.normalvariate(0, 1) for j in range(embedding_size)])
                word_num += 1

        x.append(line_x)
    
    return x, word_num, embedding_list


if __name__ == "__main__":
    train_data_path = "data/msr_training.utf8"
    test_data_path = "data/msr_test_gold.utf8"
    embedding_path = "embedding/bert_embedding.txt"
    model_path = "checkpoint/bilstm_crf_epoch_6_iters_86918.pth"
    result_path = "data/result.utf8"

    EMBEDDING_SIZE = 768
    HIDDEN_DIM = 1536

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    word_num = 0
    embedding_list = []
    with open(embedding_path, "r", encoding="utf8") as embedding:
        for line in embedding:
            id2word.append(line[0])
            word2id[line[0]] = word_num
            line_split = line[1:].strip().split()
            embedding_list.append([float(x) for x in line_split])
            word_num += 1

    x_train = []
    with open(train_data_path, "r", encoding="utf8") as train_data:
        x_train, word_num, embedding_list = load_data(train_data, x_train, word_num, embedding_list, EMBEDDING_SIZE)
    x_test = []
    with open(test_data_path, "r", encoding="utf8") as test_data:
        x_test, word_num, embedding_list = load_data(test_data, x_test, word_num, embedding_list, EMBEDDING_SIZE)

    model = BiLSTM_CRF(len(word2id)+1, tag2id, EMBEDDING_SIZE, HIDDEN_DIM, device, embedding_list).to(device)
    model.load_state_dict(torch.load(model_path))

    x_train.extend(x_test)
    f = open(result_path, "w", encoding="utf8")
    for sentence in x_train:
        with torch.no_grad():
            sentence = torch.LongTensor(sentence).to(device)
            predict = model.decode(sentence)
        
        predict_item = []
        temp_item = []
        for i in range(len(sentence)):
            if id2tag[predict[i]] == "B":
                temp_item = [i]
            elif id2tag[predict[i]] == "M" and len(temp_item) != 0:
                temp_item.append(i)
            elif id2tag[predict[i]] == "E" and len(temp_item) != 0:
                temp_item.append(i)
                predict_item.append(temp_item)
                temp_item = []
            elif id2tag[predict[i]] == "S":
                temp_item = [i]
                predict_item.append(temp_item)
                temp_item = []
            else:
                temp_item = []
        
        for item in predict_item:
            for c in item:
                f.write(id2word[sentence[c]])
            f.write(" ")
        f.write("\n")

    f.close()