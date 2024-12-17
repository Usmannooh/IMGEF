import torch
import torch.nn as nn
from torch.nn import Parameter


class GCN(nn.Module):

    def __init__(self, in_size, state_size):
        super(GCN, self).__init__()


        self.gcn1 = GraphConvolutionLayer(in_size, state_size)
        self.gcn2 = GraphConvolutionLayer(in_size, state_size)
        self.gcn3 = GraphConvolutionLayer(in_size, state_size)

    def forward(self, states, fw_A, bw_A):
        states = states.permute(0, 2, 1)
        states = self.gcn1(states, fw_A, bw_A)
        states = self.gcn2(states, fw_A, bw_A)
        states = self.gcn3(states, fw_A, bw_A)

        return states.permute(0, 2, 1)

class GCLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super().__init__()
        self.condense = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.condense_norm = nn.BatchNorm1d(state_size)
        self.fw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.fw_norm = nn.BatchNorm1d(state_size)
        self.bw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.bw_norm = nn.BatchNorm1d(state_size)
        self.update = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.update_norm = nn.BatchNorm1d(in_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, states, fw_A, bw_A):
        condensed = self.relu(self.condense_norm(self.condense(states)))
        fw_msg = self.relu(self.fw_norm(self.fw_trans(states).bmm(fw_A)))
        bw_msg = self.relu(self.bw_norm(self.bw_trans(states).bmm(bw_A)))
        updated = self.update_norm(self.update(torch.cat((condensed, fw_msg, bw_msg), dim=1)))
        updated = self.relu(updated + states)
        return updated


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super(GraphConvolutionLayer, self).__init__()

        self.in_size = in_size
        self.state_size = state_size

        self.condense = nn.Conv1d(in_size, state_size, 1)
        self.condense_norm = nn.BatchNorm1d(state_size)

        self.gcn_forward = GraphConvolution(in_size, state_size)
        self.gcn_backward = GraphConvolution(in_size, state_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.conv1d = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.norm = nn.BatchNorm1d(in_size)

        self.test_conv = nn.Conv1d(state_size, in_size, 1, bias=False)

    def forward(self, x, fw_A, bw_A):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        states = x
        condensed_message = self.relu(self.condense_norm(self.condense(x)))
        fw_message = self.relu(self.gcn_forward(x, fw_A)).to(device)
        bw_message = self.relu(self.gcn_backward(x, bw_A)).to(device)
        update = torch.cat((condensed_message, fw_message, bw_message), dim=1)
        x = self.norm(self.conv1d(update))
        x = self.relu(x + states)

        return x


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x = x.permute(0, 2, 1).to(device)
        self.weight = Parameter(torch.FloatTensor(x.size()[0], self.in_features, self.out_features).to('cuda:0'))
        self.reset_parameters_xavier()
        support = torch.bmm(x, self.weight).to(device)
        adj = adj.permute(0, 2, 1).to(device)
        output = torch.matmul(adj, support).to(device)


        if self.bias is not None:
            return (output + self.bias).permute(0, 2, 1)
        else:
            return output.permute(0, 2, 1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'