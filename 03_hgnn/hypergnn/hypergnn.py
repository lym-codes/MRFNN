from collections import defaultdict
from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.activation import PReLU
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
from torch_sparse.tensor import to
from hyperutils import *

class HyperGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_edge_num=5, num_layer=1, negative_slope=0.2):
        super(HyperGNN, self).__init__()
        self.negative_slope = negative_slope
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(hyper_edge_num, 1))  # 边的权重
        glorot(self.alpha)

    def forward(self, company_emb, hyp_graph):
        hyperlist = []
        for i in range(len(hyp_graph)):
            laplacian = scipy_sparse_mat_to_torch_sparse_tensor(hyp_graph[i].laplacian())
            com = laplacian @ self.proj(company_emb)
            hyperlist += [com]
        res = 0
        alpha = torch.sigmoid(self.alpha)
        for i in range(len(hyperlist)):
            res += hyperlist[i] * alpha[i]
        return res

class RiskInfo(nn.Module):
    # def __init__(self, input_dim, company_num,audit_type_num):
    def __init__(self, input_dim, company_num):
        super(RiskInfo, self).__init__()
        self.input_dim = input_dim
        self.company_num = company_num
        self.basic =16

    def forward(self, risk_data):
        # 定义embedding编码审计类型 aduit_type_num=6
        com_emb = torch.zeros((self.company_num, self.basic))
        # print(len(risk_data))
        for line in risk_data:
            basic_data = line[1:16]+[17]  # 15+1=16维基础信息
            basic_data = torch.FloatTensor(basic_data)
            com_emb[int(line[16])] = basic_data
        return com_emb

from torch_geometric.nn import RGATConv
from torch import Tensor, LongTensor
class RGATModel(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, hidden_l: int, num_rels: int, num_classes: int) -> None:
        super(RGATModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.rgat1 = RGATConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_rels, num_bases=None,concat=False,heads=3)
        self.rgat2 = RGATConv(in_channels=hidden_l, out_channels=num_classes, num_relations=num_rels,
                              num_bases=None,concat=False,heads=3)
        # intialize weights
        nn.init.kaiming_uniform_(self.rgat1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.rgat2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, edge_index: LongTensor, edge_type: LongTensor) -> Tensor:
        x1 = self.rgat1(self.embedding.weight, edge_index, edge_type)
        x2 = torch.relu(x1)
        x3 = self.rgat2(x2, edge_index, edge_type)
        x4 = torch.sigmoid(x3)
        return x4

from graph import Graph



'''
LSTM
'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Take the last LSTM output
        output = self.fc(output)
        return output

'''
GRU
'''
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        output = output[:, -1, :]  # Take the last GRU output
        output = self.fc(output)
        return output




class RiskGNN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, device, com_initial_emb, node_num, rel_num, num_class, num_heads=1, dropout=0.25, norm=True):
        super(RiskGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.company_num = company_num
        self.device = device
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm = norm
        self.info_proj = nn.Linear(output_dim, output_dim, bias=False)
        self.company_emb = torch.FloatTensor(com_initial_emb)  # 训练的企业向量
        self.riskinfo = RiskInfo(input_dim, company_num)
        # 02 hyper
        self.hypergnn = HyperGNN(input_dim, output_dim, num_layer=1)
        # 03 rgat
        emb_dim = 50
        hidden_l = 4
        self.rgat = RGATModel(node_num,emb_dim,hidden_l,rel_num, num_class)
        self.company_proj = nn.Linear(32, input_dim, bias=False)
        self.risk_proj = nn.Linear(input_dim + 16, input_dim, bias=False)
        self.final_proj = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False), nn.ReLU(), nn.Linear(output_dim, output_dim, bias=False))
        self.alpha = torch.ones((1))
        self.weight = torch.ones((1))
        # 04 lstm
        self.event_vec_input=1
        self.event_vec_ouput=1
        self.eventlstm= LSTMModel(1,8,1) #inupt;hidden;output
        # 05 gru
        self.eventgru = GRUModel(1, 8, 1)  # inupt;hidden;output

    # def forward(self, risk_data, hyp_graph, idx,x,edge_index,edge_type):
    def forward(self, risk_data, idx,edge_index,edge_type,risk_event):
        # legal_rank = np.asanyarray(risk_data)[1:16]+[17]
        # legal_rank = torch.FloatTensor(legal_rank)
        # risk_data,15维基础向量 idx,企业id，edge_index,edge_type,risk_event，事件融合
        #0、企业编号
        id_index = torch.tensor([int(line[16]) for line in risk_data])
        #1、 基础向量
        company_emb = self.company_proj(self.company_emb)  # 格式化为输入的的企业向量
        risk_info = self.riskinfo(risk_data)  # 拼接 riskdata
        # 00 消融hyper，只保留node2vec
        risk_node2vec = self.risk_proj(torch.cat((company_emb, risk_info), dim=1))
        # 01 消融nodevec，只保留hyper
        # risk_hyper = self.hypergnn(risk_info,hyp_graph)
        # 02 全都保留，risk+node2vec+hyper
        # risk_node2vec_hyper = self.hypergnn(risk_node2vec, hyp_graph)

        #2、rgat
        node_index=torch.tensor([int(line[-1]) for line in risk_data])
        rgat=self.rgat(edge_index,edge_type)
        rgat_emb=rgat[node_index] # 含数据的部分
        rgat_emb_final = torch.zeros((self.company_num, 2))
        rgat_emb_final[id_index] = rgat_emb  # 完整部分

        #3、一阶拼接
        # alpha = torch.sigmoid(self.alpha)
        # 00 消融hyper,只保留node2vec
        first_mer =  self.final_proj(risk_node2vec)
        # 01 消融node2vec,只保留hyper
        # first_mer = alpha * F.gelu(risk_hyper) + (1 - alpha) * self.final_proj(risk_info)
        # 02 全都保留，risk+node2vec+hyper
        # first_mer = alpha * F.gelu(risk_node2vec_hyper) + (1 - alpha) * self.final_proj(risk_node2vec)
        # 03 全消融，只保留最原始的risk
        # first_mer = self.final_proj(risk_info)

        # 4、二阶拼接
        # weight= torch.sigmoid(self.weight)
        sec_mer = torch.cat((first_mer, rgat_emb_final), dim=1)

        # 5. 三阶拼接
        event_emb_input=self.event_vec_input
        event_emb_output=self.event_vec_ouput
        # event_vec=self.eventlstm(risk_event[:, :, :event_emb_input])
        event_vec = self.eventgru(risk_event[:, :, :event_emb_input])
        event_vec_final = torch.zeros((self.company_num, event_emb_output))
        event_vec_final[id_index] = event_vec  # 完整部分
        third_mer=torch.cat((sec_mer, event_vec_final), dim=1)
        return third_mer[idx]


