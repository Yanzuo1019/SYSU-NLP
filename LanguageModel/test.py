import argparse

import torch
import torch.nn as nn

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 1024

data_path = "data/result.utf8"

word2id = {}
id2word = []
word_num = 0

word2id["<PAD>"] = word_num
id2word.append("<PAD>")
word_num += 1

word2id["<EOS>"] = word_num
id2word.append("<EOS>")
word_num += 1

sentences = []
seq_lens = []

device = None

class MaskedLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor, seq_lens, hidden=None):
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        x_packed = nn.utils.rnn.pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        if hidden is None:
            y_lstm, hidden = self.lstm(x_packed)
        else:
            y_lstm, hidden = self.lstm(x_packed, hidden)
        y_padded, length = nn.utils.rnn.pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        return y_padded, hidden


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(RNNLM, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = MaskedLSTM(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, sentences, seq_lens):
        batch_size, time_step = sentences.shape
        embedding = self.word_embed(sentences)
        out, _ = self.lstm(embedding, seq_lens)
        out = self.linear(out.view(batch_size * time_step, -1))
        return out
    
    def test(self, word):
        sentence = []
        sentence.append(word2id[word])

        softmax = nn.Softmax(dim=1)

        tensor = torch.LongTensor([sentence[-1]]).view(1, -1).to(device)
        embedding = self.word_embed(tensor).view(1, 1, -1)
        out, hidden = self.lstm(embedding, [1])
        out = self.linear(out.view(1, -1))
        prob = softmax(out.view(1, -1))
        
        while torch.argmax(prob).item() != word2id["<EOS>"]:
            sentence.append(torch.argmax(prob).item())
            # print([id2word[i] for i in sentence])

            tensor = torch.LongTensor([sentence[-1]]).view(1, -1).to(device)
            embedding = self.word_embed(tensor).view(1, 1, -1)
            out, hidden = self.lstm(embedding, [1], hidden)
            out = self.linear(out.view(1, -1))
            prob = softmax(out.view(1, -1))
        
        return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
        help="path of pretrained model")
    args = parser.parse_args()

    with open(data_path, "r", encoding="utf8") as data:
        for line in data:
            line_split = line.strip().split()

            sen = []
            for word in line_split:
                if word not in word2id:
                    word2id[word] = word_num
                    id2word.append(word)
                    word_num += 1
                sen.append(word2id[word])
            
            sen.append(word2id["<EOS>"])
            sentences.append(sen)
            seq_lens.append(len(sen))
    
    vocab_size = len(word2id)
    model = RNNLM(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE)
    model.load_state_dict(torch.load(args.model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)

    while True:
        word = input("Please input the first word to generate sentence: ")
        if word not in word2id:
            print("Sorry, word {} is not in the dictionary!".format(word))
        else:
            print("Get id from array word2id: {}".format(word2id[word]))
        
        sentence = model.test(word)
        output_str = ""
        for i in range(len(sentence)):
            output_str += id2word[sentence[i]]
        
        print(output_str, "\n")