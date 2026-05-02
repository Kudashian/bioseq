import torch

class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        #Define lstm layer
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        
        #classification head
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, seq_t):
        #forward pass through LSTM cell
        o_t, (h_t, c_t) = self.lstm(seq_t, None)
        #classification head
        logits = self.classifier(h_t.squeeze(0))
        out = self.out(logits)
        return out