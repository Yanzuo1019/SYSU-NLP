import time

import torch
import torch.nn as nn

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCH = 10
BATCH_SIZE = 16

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


class MaskedLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor, seq_lens):
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        x_packed = nn.utils.rnn.pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, hidden = self.lstm(x_packed)
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


def padding(sentences, max_len):
    batch = []
    index = []
    for i, sen in enumerate(sentences):
        tensor = sen.copy()
        tensor.extend([word2id["<PAD>"]] * (max_len - len(sen)))
        tensor = torch.LongTensor(tensor)
        batch.append(tensor)
        index.extend([j for j in range(i * max_len, i * max_len + len(sen))])
    return torch.stack(batch), index


if __name__ == "__main__":
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
            
            # sen.append(word2id["<EOS>"])
            sentences.append(sen)
            seq_lens.append(len(sen))

    vocab_size = len(word2id)
    model = RNNLM(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1, verbose=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start = time.time()
    for epoch in range(EPOCH):
        total_loss = 0.0
        count = 0
        for iters, i in enumerate(range(0, len(sentences), BATCH_SIZE)):
            if len(sentences) - i > BATCH_SIZE:
                batch = sentences[i:i+BATCH_SIZE]
                lens = seq_lens[i:i+BATCH_SIZE]
            else:
                batch = sentences[i:]
                lens = seq_lens[i:]
            
            # lens = torch.LongTensor(lens).to(device)
            stcs, index = padding(batch, max(lens))
            stcs = stcs.to(device)
            out = model(stcs, lens)

            gt = torch.cat([stcs.view(-1)[1:], torch.LongTensor([word2id["<EOS>"]])])
            loss = criterion(out[index], gt)
            total_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1

        average_loss = total_loss / count
        elapsed_time = int(time.time() - start)
        print("Epoch {}/{} Average Loss: {:.6f} Elapsed Time: {}m{}s".format(epoch + 1, EPOCH, average_loss, elapsed_time // 60, elapsed_time % 60))
        scheduler.step(average_loss)
        torch.save(model.state_dict(), "checkpoint/rnnlm_epoch_{}.pth".format(epoch + 1))
