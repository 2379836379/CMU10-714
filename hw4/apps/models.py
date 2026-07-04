import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn



def ConvBN(in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
    return nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype),
        nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
        nn.ReLU(),
    )


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.model = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(nn.Sequential(
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
            )),
            ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(nn.Sequential(
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
            )),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.model(x)


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1, seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        emb = self.embedding(x)
        out, h = self.seq_model(emb, h)
        seq_len, bs, hidden = out.shape
        out = self.linear(out.reshape((seq_len * bs, hidden)))
        return out, h
