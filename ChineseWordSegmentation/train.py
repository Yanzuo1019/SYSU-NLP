# Author: Lu Yanzuo
# Date: 2020-11-26
# Description: CWS BiLSTM-CRF Model

import os
import logging
import argparse
import random

import torch
import torch.optim as optim

from model import BiLSTM_CRF

id2tag = ['B', 'M', 'E', 'S']
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
id2word = []
word2id = {}

START_TAG = "<START>"
STOP_TAG = "<STOP>"
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
        # print("sentence", sentence)
        # print("tags", tags)
        # print("predict", predict)

        predict_item = []
        gt_item = []

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

        # print(predict_item)

        temp_item = []
        for i in range(len(sentence)):
            if id2tag[tags[i]] == "B":
                temp_item = [i]
            elif id2tag[tags[i]] == "M" and len(temp_item) != 0:
                temp_item.append(i)
            elif id2tag[tags[i]] == "E" and len(temp_item) != 0:
                temp_item.append(i)
                gt_item.append(temp_item)
                temp_item = []
            elif id2tag[tags[i]] == "S":
                temp_item = [i]
                gt_item.append(temp_item)
                temp_item = []
            else:
                gt_item = []
        
        # print(gt_item)
    
        correct_pred = [i for i in predict_item if i in gt_item]
        pred_num += len(predict_item)
        gt_num += len(gt_item)
        correct_pred_num += len(correct_pred)

        # print("correct_pred", correct_pred)
        # exit()

    if correct_pred_num > 0:
        precision = float(correct_pred_num) / float(pred_num)
        recall = float(correct_pred_num) / float(gt_num)
        F1 = (2 * precision * recall) / (precision + recall)
        logger.info("Test Result --- precision = {:.4f}".format(precision))
        logger.info("Test Result --- recall = {:.4f}".format(recall))
        logger.info("Test Result --- F1 score = {:.4f}".format(F1))
        return F1
    else:
        logger.info("Test Result --- F1 score = {:.4f}".format(0.0))
        return 0.0

def load_data(data, x, y, word_num, embedding_list, embedding_size):
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

        line_split = line.split()
        line_y = []
        for item in line_split:
            line_y.extend(item2tag(item))

        x.append(line_x)
        y.append(line_y)
    
    return x, y, word_num, embedding_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--console", action="store_true", 
        help="print log to the console instead of logfile")
    parser.add_argument("--only_test", action="store_true",
        help="skip training and only test")
    parser.add_argument("--model_path", type=str, default=None, 
        help="pretrain model path")
    parser.add_argument("--embedding_path", type=str, default=None,
        help="pretrain word embedding")
    parser.add_argument("--train_data_path", type=str, default="data/msr_training.utf8",
        help="train data path")
    parser.add_argument("--test_data_path", type=str, default="data/msr_test_gold.utf8",
        help="test data path (with groundtruth but not raw data)")
    parser.add_argument("--embedding_size", type=int, default=768, 
        help="word embedding size")
    parser.add_argument("--hidden_dim", type=int, default=1536, 
        help="BiLSTM hidden layer dim")
    parser.add_argument("--epoch", type=int, default=3, 
        help="training epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
        help="training learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
        help="optimizer weight decay rate")
    parser.add_argument("--start_epoch", type=int, default=0,
        help="epoch number inherit from pretrain model")
    args = parser.parse_args()

    console = args.console
    only_test = args.only_test
    model_path = args.model_path
    embedding_path = args.embedding_path
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_DIM = args.hidden_dim
    EPOCH = args.epoch
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    START_EPOCH = args.start_epoch

    os.makedirs("checkpoint", exist_ok=True)
    logger = logging.getLogger()
    if console:
        logging.basicConfig(
            level=logging.DEBUG, 
            datefmt="%Y-%m-%d %H:%M:%S",
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
        )
    else:
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
    embedding_list = None
    if embedding_path is not None:
        embedding_list = []
        with open(embedding_path, "r", encoding="utf8") as embedding:
            for line in embedding:
                id2word.append(line[0])
                word2id[line[0]] = word_num
                line_split = line[1:].strip().split()
                embedding_list.append([float(x) for x in line_split])
                word_num += 1

    x_train = []
    y_train = []
    logger.info("Reading train data")
    with open(train_data_path, "r", encoding="utf8") as train_data:
        x_train, y_train, word_num, embedding_list = load_data(train_data, x_train, y_train, word_num, embedding_list, EMBEDDING_SIZE)
    logger.info("Read train data successfully")
    
    x_test = []
    y_test = []
    logger.info("Reading test data.")
    with open(test_data_path, "r", encoding="utf8") as test_data:
        x_test, y_test, word_num, embedding_list = load_data(test_data, x_test, y_test, word_num, embedding_list, EMBEDDING_SIZE)
    logger.info("Read test data successfully")
    
    model = BiLSTM_CRF(len(word2id)+1, tag2id, EMBEDDING_SIZE, HIDDEN_DIM, device, embedding_list).to(device)
    optimizer = optim.SGD(filter(lambda para: para.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=1, verbose=True)
    logger.info("Model and optimizer initialize successfully")

    f1_score = test(x_test, y_test)

    if not only_test:
        logger.info("Start training")
        for epoch in range(EPOCH):
            iters = 0
            for sentence, tags in zip(x_train, y_train):
                iters += 1

                sentence = torch.LongTensor(sentence).to(device)
                tags = torch.LongTensor(tags).to(device)

                log_sum_exp_score, gt_score = model(sentence, tags)
                loss = log_sum_exp_score - gt_score

                model.zero_grad()
                loss.backward()
                optimizer.step()

                if iters % 10000 == 0:
                    logger.info("epoch {}/{} iters {}/{}".format(epoch+1, EPOCH, iters, len(x_train)))
                    f1_score = test(x_test, y_test)
                    scheduler.step(f1_score)
                    torch.save(model.state_dict(), "checkpoint/bilstm_crf_epoch_{}_iters_{}.pth".format(epoch+START_EPOCH+1, iters))
                    logger.info("save model to path {}".format("checkpoint/bilstm_crf_epoch_{}_iters_{}.pth".format(epoch+START_EPOCH+1, iters)))
                elif iters % 1000 == 0:
                    logger.info("epoch {}/{} iters {}/{}".format(epoch+1, EPOCH, iters, len(x_train)))

            f1_score = test(x_test, y_test)
            scheduler.step(f1_score)
            torch.save(model.state_dict(), "checkpoint/bilstm_crf_epoch_{}_iters_{}.pth".format(epoch+START_EPOCH+1, iters))
            logger.info("save model to path {}".format("checkpoint/bilstm_crf_epoch_{}_iters_{}.pth".format(epoch+START_EPOCH+1, iters)))
        logger.info("Finish training")