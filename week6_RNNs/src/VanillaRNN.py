import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        # define W_xh and W_hh here
        self.w_xh = torch.nn.Linear(input_size, hidden_size)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        # compute h_t using both x_t and h_prev
        # return h_t
        h_t = torch.tanh(self.w_xh(x_t) + self.w_hh(h_prev))
        # update h_prev
        h_prev = h_t
        return h_prev, h_t