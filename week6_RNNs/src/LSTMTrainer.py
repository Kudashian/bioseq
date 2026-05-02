import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        #define weights for each gate and transpose for x_t and h_prev
        self.w_ih = torch.nn.Linear(input_size, hidden_size)
        self.w_fh = torch.nn.Linear(input_size, hidden_size)
        self.w_oh = torch.nn.Linear(input_size, hidden_size)
        self.wgh = torch.nn.Linear(input_size, hidden_size)
        self.w_ihh = torch.nn.Linear(hidden_size, hidden_size)
        self.w_fhh = torch.nn.Linear(hidden_size, hidden_size)
        self.w_ohh = torch.nn.Linear(hidden_size, hidden_size)
        self.wghh = torch.nn.Linear(hidden_size, hidden_size)
        #classification head
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, seq_t, c_prev, h_prev):
        for x_t in seq_t:
        #Compute i_t, f_t, o_t, g_t
            i_t = torch.sigmoid(self.w_ih(x_t) + self.w_ihh(h_prev))
            f_t = torch.sigmoid(self.w_fh(x_t) + self.w_fhh(h_prev))
            o_t = torch.sigmoid(self.w_oh(x_t) + self.w_ohh(h_prev))
            g_t = torch.tanh(self.wgh(x_t) + self.wghh(h_prev))
        #add contribution from h_prev
            c_t = f_t * c_prev + i_t * g_t
        #update h_t, h_prev and c_prev for next iteration
            h_t = o_t * torch.tanh(c_t)
            c_prev = c_t
            h_prev = h_t

        #classification head
        logits = self.classifier(h_t)
        out = self.out(logits)
        return c_t, h_t, out