from torch import nn

class RNN(nn.Module):

    def __init__ (self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, input_tensor, prev_hidden=None):
        hidden_tensors, final_hidden = self.rnn(input_tensor, prev_hidden)

        #.fc takes last hidden dim output and predicts output
        prediction = self.fc(hidden_tensors[:, -1, :]) #takes LAST TIMESTAMP OF r_out
        return prediction, final_hidden