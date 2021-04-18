import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_labels):
        super(Net, self).__init__()
        self.numl = num_labels
        self.model_name = "rcnn"
        self.fc1 = nn.Linear(192, 320)
        self.bilstm = nn.LSTM(input_size=320, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(128*2, 22)

    def forward(self, x):        
        xs = x.size()
        out = x.view(xs[0], xs[1], xs[2]).permute(0, 2, 1)
        out = self.fc1(out).contiguous()
        out, _ = self.bilstm(out)
        out = self.fc2(out)
        out = out.view(out.shape[0], 1, out.shape[1], out.shape[2]).permute(0, 1, 3, 2)
        return out
