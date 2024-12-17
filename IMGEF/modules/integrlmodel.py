import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Parameter


class IntegratedModel(nn.Module):
    def __init__(self, input_dims, output_dim, fusion_type="con", dropout_rate=0.3, in_feats=2048, out_feats=512):  # in_feats=2048, out_feats=512
        super(IntegratedModel, self).__init__()

        self.multi_modal_fusion = MultiModalFeatFusion(input_dims, output_dim, fusion_type, dropout_rate)
        self.sgem = SptialAware(in_feats, out_feats)
        self.graph = Sgemlayer(in_feats, out_feats)
    def forward(self, image_feats, text_feats, graph_feats, node_feats, neighbor_feats, adjacency_matrix):
        fused_feats = self.multi_modal_fusion(image_feats, text_feats, graph_feats)
        sgem_output = self.sgem(node_feats, neighbor_feats)
        graph_output = self.graph(sgem_output, adjacency_matrix)
        combined_output = fused_feats

        return combined_output


class MultiModalFeatFusion(nn.Module):
    def __init__(self, input_dims, output_dim, fusion_type="con", dropout_rate=0.3):
        super(MultiModalFeatFusion, self).__init__()
        self.fusion_type = fusion_type


        self.image_proj = nn.Linear(input_dims['image'], output_dim)
        self.text_proj = nn.Linear(input_dims['text'], output_dim)
        self.graph_proj = nn.Linear(input_dims['graph'], output_dim)
        self.image_bn = nn.BatchNorm1d(output_dim)
        self.text_bn = nn.BatchNorm1d(output_dim)
        self.graph_bn = nn.BatchNorm1d(output_dim)
        if self.fusion_type == "con":
            self.fusion_layer = nn.Linear(output_dim * 3, output_dim)
        elif self.fusion_type == "sm":
            self.fusion_layer = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, image_feats, text_feats, graph_feats):
        img_emb = self.image_bn(self.image_proj(image_feats))
        txt_emb = self.text_bn(self.text_proj(text_feats))
        graph_emb = self.graph_bn(self.graph_proj(graph_feats))
        img_emb, _ = self.attention_layer(img_emb.unsqueeze(0), img_emb.unsqueeze(0), img_emb.unsqueeze(0))
        txt_emb, _ = self.attention_layer(txt_emb.unsqueeze(0), txt_emb.unsqueeze(0), txt_emb.unsqueeze(0))
        graph_emb, _ = self.attention_layer(graph_emb.unsqueeze(0), graph_emb.unsqueeze(0), graph_emb.unsqueeze(0))


        if self.fusion_type == "con":
            fused_feats = torch.cat([img_emb.squeeze(0), txt_emb.squeeze(0), graph_emb.squeeze(0)], dim=-1)
        elif self.fusion_type == "ad":
            fused_feats = img_emb.squeeze(0) + txt_emb.squeeze(0) + graph_emb.squeeze(0)

        fused_feats = self.fusion_layer(fused_feats)
        fused_feats = self.activation(self.dropout(fused_feats))

        return fused_feats

class SptialAware(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='lm', use_bias=True):
        super(SptialAware, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        self.weight_matrix = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        if self.aggregator_type == 'ama':
            self.aggregator = self.mean_aggregator
        elif self.aggregator_type == 'dfa':
            self.pool_fc = nn.Linear(in_feats, in_feats)
            self.aggregator = self.pool_aggregator
        elif self.aggregator_type == 'lm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
            self.aggregator = self.lstm_aggregator

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.batch_norm = nn.BatchNorm1d(out_feats)


        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, node_feats, neighbor_feats, adjacency_list=None):
        device = self.weight_matrix.device
        node_feats = node_feats.to(device)
        neighbor_feats = neighbor_feats.to(device)
        aggregated_feats = self.aggregator(neighbor_feats)
        combined_feats = torch.cat([node_feats, aggregated_feats], dim=-1)
        convolved_feats = torch.matmul(combined_feats, self.weight_matrix) # Apply transformation with learned weight matrix

        if self.bias is not None:
            convolved_feats += self.bias
        convolved_feats = self.batch_norm(convolved_feats)

        return convolved_feats
    def mn_aggregator(self, neighbor_feats):
        return neighbor_feats.mean(dim=1)
    def pl_aggregator(self, neighbor_feats):
        neighbor_feats = F.relu(self.pool_fc(neighbor_feats))
        return neighbor_feats.max(dim=1)[0]
    def lm_aggregator(self, neighbor_feats):
        neighbor_feats, _ = self.lstm(neighbor_feats)
        return neighbor_feats.mean(dim=1)

class Sgemlayer(nn.Module):
    def __init__(self, in_feats, out_feats, use_bias=True):
        super(Sgemlayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight_matrix = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.batch_norm = nn.BatchNorm1d(out_feats)
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feats))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.xavier_normal_(self.weight_matrix, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input_feats, adjacency_matrix):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = self.weight_matrix.device
        input_feats = input_feats.to(device)
        adjacency_matrix = adjacency_matrix.to(device)
        input_feats = input_feats.permute(0, 2, 1)  # Shape: (batch_size, feats, seq_len)
        support_matrix = torch.bmm(input_feats, self.weight_matrix.unsqueeze(0).expand(input_feats.size(0), -1, -1))
        convolved_feats = torch.bmm(adjacency_matrix, support_matrix)
        convolved_feats = self.batch_norm(convolved_feats)  # Apply batch normalization here

        if self.bias is not None:
            convolved_feats += self.bias

        return convolved_feats.permute(0, 2, 1)

