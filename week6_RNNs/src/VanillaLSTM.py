import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        #define weights for each gate
        self.w_ih = torch.nn.Linear(input_size, hidden_size)
        self.w_fh = torch.nn.Linear(input_size, hidden_size)
        self.w_oh = torch.nn.Linear(input_size, hidden_size)
        self.wgh = torch.nn.Linear(input_size, hidden_size)
        self.w_ihh = torch.nn.Linear(hidden_size, hidden_size)
        self.w_fhh = torch.nn.Linear(hidden_size, hidden_size)
        self.w_ohh = torch.nn.Linear(hidden_size, hidden_size)
        self.wghh = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, c_prev, h_prev):
        #Compute i_t, f_t, o_t, g_t
        i_t = torch.sigmoid(self.w_ih(x_t) + self.w_ihh(h_prev))
        f_t = torch.sigmoid(self.w_fh(x_t) + self.w_fhh(h_prev))
        o_t = torch.sigmoid(self.w_oh(x_t) + self.w_ohh(h_prev))
        g_t = torch.tanh(self.wgh(x_t) + self.wghh(h_prev))
        #add contribution from h_prev
        c_t = f_t * c_prev + i_t * g_t
        #update h_t
        h_t = o_t * torch.tanh(c_t)
        return c_t, h_t