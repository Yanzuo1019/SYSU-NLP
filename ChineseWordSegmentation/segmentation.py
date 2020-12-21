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


def load_data(data, x, word_num):
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
                word_num += 1

        x.append(line_x)
    
    return x, word_num


if __name__ == "__main__":
    train_data_path = ""
    test_data_path = ""
    EMBEDDING_SIZE = 768
    HIDDEN_DIM = 1536
    model_path = ""
    result_path = ""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    word_num = 0
    x_train = []
    with open(train_data_path, "r", encoding="utf8") as train_data:
        x_train, word_num = load_data(train_data, x_train, word_num)
    x_test = []
    with open(test_data_path, "r", encoding="utf8") as test_data:
        x_test, word_num = load_data(test_data, x_test, word_num)
    
    model = BiLSTM_CRF(len(word2id)+1, tag2id, EMBEDDING_SIZE, HIDDEN_DIM, device, None).to(device)
    model.load_state_dict(torch.load(model_path))

    x = x_train.extend(x_test)
    f = open(result_path, "w", encoding="utf8")
    for sentence in x:
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
                f.write(id2word[c])
            f.write(" ")
        f.write("\n")

    f.close()