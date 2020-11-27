# Author: Lu Yanzuo
# Date: 2020-11-26
# Description: CWS BiLSTM-CRF Model

import logging

import torch
import torch.optim as optim

from model import BiLSTM_CRF

train_data_path = "data/msr_training.utf8"
test_data_path = "data/msr_test_gold.utf8"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_SIZE = 100
HIDDEN_DIM = 200
EPOCH = 5
LEARNING_RATE = 0.005

id2tag = ['B', 'M', 'E', 'S']
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
id2word = []
word2id = {}
tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)

def item2tag(item):
    result = []
    if len(item) == 1:
        result.append(tag2id['S'])
    elif len(item) == 2:
        result.extend([tag2id['B'], tag2id['E']])
    else:
        M_num = len(item) - 2
        M_list = [tag2id['M']] * M_num
        result.append(tag2id['B'])
        result.extend(M_list)
        result.append(tag2id['E'])
    return result

def test(x_test, y_test):
    logger.info("Testing...Wait!")

    pred_num = 0
    gt_num = 0
    correct_pred_num = 0

    for sentence, tags in zip(x_test, y_test):
        with torch.no_grad():
            sentence = torch.LongTensor(sentence).to(device)
            predict = model.decode(sentence)

        predict_item = []
        gt_item = []

        temp_item = []
        for i in range(len(sentence)):
            if id2tag[predict[i]] == "B":
                temp_item = [id2word[sentence[i]]]
            elif id2tag[predict[i]] == "M" and len(temp_item) != 0:
                temp_item.append(id2word[sentence[i]])
            elif id2tag[predict[i]] == "E" and len(temp_item) != 0:
                temp_item.append(id2word[sentence[i]])
                predict_item.append(temp_item)
                temp_item = []
            elif id2tag[predict[i]] == "S":
                temp_item = [id2word[sentence[i]]]
                predict_item.append(temp_item)
                temp_item = []
            else:
                temp_item = []

        temp_item = []
        for i in range(len(sentence)):
            if id2tag[tags[i]] == "B":
                temp_item = [id2word[sentence[i]]]
            elif id2tag[tags[i]] == "M" and len(temp_item) != 0:
                temp_item.append(id2word[sentence[i]])
            elif id2tag[tags[i]] == "E" and len(temp_item) != 0:
                temp_item.append(id2word[sentence[i]])
                gt_item.append(temp_item)
                temp_item = []
            elif id2tag[tags[i]] == "S":
                temp_item = [id2word[sentence[i]]]
                gt_item.append(temp_item)
                temp_item = []
            else:
                temp_item = []
    
        correct_pred = [i for i in predict_item if i in gt_item]
        pred_num += len(predict_item)
        gt_num += len(gt_item)
        correct_pred_num += len(correct_pred)

    if correct_pred_num > 0:
        precision = float(correct_pred_num) / float(pred_num)
        recall = float(correct_pred_num) / float(gt_num)
        F1 = (2 * precision * recall) / (precision + recall)
        logger.info("Test Result --- precision = {:.2f}".format(precision))
        logger.info("Test Result --- recall = {:.2f}".format(recall))
        logger.info("Test Result --- F1 score = {:.2f}".format(F1))
    else:
        logger.info("Test Result --- F1 score = {:.2f}".format(0.0))

if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.DEBUG, 
        datefmt="%Y-%m-%d %H:%M:%S",
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
        filemode="w",
        filename="train.log"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(str(device)))

    word_num = 0

    x_train = []
    y_train = []
    logger.info("Reading train data")
    with open(train_data_path, "r", encoding="utf8") as train_data:
        for line in train_data:
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

            line_split = line.split()
            line_y = []
            for item in line_split:
                line_y.extend(item2tag(item))

            x_train.append(line_x)
            y_train.append(line_y)
    logger.info("Read train data successfully")
    
    x_test = []
    y_test = []
    logger.info("Reading test data.")
    with open(test_data_path, "r", encoding="utf8") as test_data:
        for line in test_data:
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

            line_split = line.split()
            line_y = []
            for item in line_split:
                line_y.extend(item2tag(item))

            x_test.append(line_x)
            y_test.append(line_y)
    logger.info("Read test data successfully")
    
    model = BiLSTM_CRF(len(word2id)+1, tag2id, EMBEDDING_SIZE, HIDDEN_DIM, device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    logger.info("Model and optimizer initialize successfully")

    test(x_test, y_test)

    logger.info("Start training")
    for epoch in range(EPOCH):
        iters = 0
        for sentence, tags in zip(x_train, y_train):
            iters += 1
            model.zero_grad()

            sentence = torch.LongTensor(sentence).to(device)
            tags = torch.LongTensor(tags).to(device)

            log_sum_exp_score, gt_score = model(sentence, tags)
            loss = log_sum_exp_score - gt_score

            loss.backward()
            optimizer.step()

            if iters % 10000 == 0:
                logger.info("epoch {}/{} iters {}/{}".format(epoch+1, EPOCH, iters, len(x_train)))
                test(x_test, y_test)
                torch.save(model.state_dict(), "checkpoint/bilstm_crf_epoch_{}_iters_{}.pth".format(epoch+1, iters))

        test(x_test, y_test)
        torch.save(model.state_dict(), "checkpoint/bilstm_crf_epoch_{}_iters_{}.pth".format(epoch+1, iters))