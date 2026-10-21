from torch import nn

class RNN(nn.Module):

    def __init__ (self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

        #data clensing search up

    def forward(self, x, hidden=None):
        # shapes RNN output to r_out and h_out, shaped by x sizing
        # r_out: shape of (batch, seq_len, hidden_dim), final layery hidden at each timestamp
        r_out, h_out = self.rnn(x, hidden)

        #.fc takes last hidden dim output and predicts output
        out = self.fc(r_out[:, -1, :]) #takes LAST TIMESTAMP OF r_out
        return out, h_out