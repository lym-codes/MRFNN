# coding: utf-8
import pickle

from matplotlib import pyplot as plt

from hypergnn import RiskGNN
from hyperutils import *
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import gmean
import time
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_kernels
import math

'''
  01 参数定义
'''
# 数据集参数
parser = argparse.ArgumentParser(description='Training GNN')

parser.add_argument('--data_dir', type=str, default='./data',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='.\model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
# 模型参数
parser.add_argument('--conv_name', type=str, default='riskgnn',
                    choices=['riskgnn'],
                    help='The name of GNN filter.')
parser.add_argument('--input_dim', type=int, default=16,
                    help='Number of input dimension')
parser.add_argument('--output_dim', type=int, default=12,
                    help='Number of output dimension')
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of HeteGAT layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
# 优化参数
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--n_epoch', type=int, default=60,
                    help='Number of epoch to run')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay of adamw ')

'''
  02 数据集划分
'''
year = '2018'
fin_file = './00_basic_data/02_basic/' + year + '.xlsx'
alldata = pd.read_excel(fin_file)  # 带风险特征与利益社区特征
alldata['证券代码'] = alldata['证券代码'].astype(str).str.zfill(6)
data = alldata.drop(columns=['违规标签'])  # 除了label都包括
label = alldata['违规标签'].tolist()
# a = label[:-200]
mix_risk_data, test_risk_data, mix_label, test_label = train_test_split(data, label,test_size=0.2,
                                                                        random_state=1)  # 划分 train与test
train_risk_data, valid_risk_data, train_label, valid_label = train_test_split(mix_risk_data, mix_label, test_size=0.25,
                                                                              random_state=2)  # 划分 train与valid
y = 0
for i in train_label:
    if i == 1:
        y+=1
z = 0
for i in test_label:
    if i == 1:
        z+=1
# train_risk_data = pd.concat([train_risk_data, fakedata], axis=0)
#111
train_idx = (train_risk_data['证券代码']).tolist()
train_id = (train_risk_data['id']).tolist()
# x_train = train_risk_data.drop(columns=['证券代码', 'id'])
# x_train = x_train.values.tolist()
train_risk_data = train_risk_data.values.tolist()
# train_label = np.append(train_label, np.ones(300))
#111


valid_idx = (valid_risk_data['证券代码']).tolist()
valid_id = (valid_risk_data['id']).tolist()
# x_valid = valid_risk_data.drop(columns=['证券代码', 'id'])
# x_valid = x_valid.values.tolist()
valid_risk_data = valid_risk_data.values.tolist()

test_idx = (test_risk_data['证券代码']).tolist()
test_id = (test_risk_data['id']).tolist()
# x_test = test_risk_data.drop(columns=['证券代码', 'id'])
# x_test = x_test.values.tolist()
test_risk_data = test_risk_data.values.tolist()

total_company_num = len(data)
train_company_num = len(train_idx)
valid_company_num = len(valid_idx)
test_company_num = len(test_idx)
# audit_type = 6

'''
  03 风险事件
'''
evdf = pd.read_excel("./02_timerisk/事件融合/" + year + ".xlsx")
evdf['公司代码'] = evdf['公司代码'].astype(str).str.zfill(6)
flat_data = evdf.values.tolist()
event_list = []
current_company = None
company_events = []
for event in flat_data:
    company_code = event[0]
    if company_code != current_company:
        if company_events:
            event_list.append(company_events)
        company_events = []
        current_company = company_code
    company_events.append(event)
if company_events:  # 将每个企业的时序事件添加入列表
    event_list.append(company_events)


# 3.1 填补空白事件的企业数据结构
def fill_missing_companies(data, all_companies):
    company_data_dict = {company_data[0][0]: company_data for company_data in data}
    # 填充缺失的企业代码
    filled_data = []
    for company_code in all_companies:
        if company_code in company_data_dict:
            filled_data.append(company_data_dict[company_code])
        else:
            filled_data.append([[company_code, 0, 0, 0]])
    return filled_data


all_companies = alldata.iloc[:, 0].tolist()
filled_data = fill_missing_companies(event_list, all_companies)


# 3.2 抽取对应企业的数据
def extract_companies(data, companies_list):
    extracted_data = []
    for company in companies_list:
        for event_sequence in data:
            if event_sequence[0][0] == company:
                extracted_data.append(event_sequence)
                break
    return extracted_data


train_events = extract_companies(filled_data, train_idx)
valid_events = extract_companies(filled_data, valid_idx)
test_events = extract_companies(filled_data, test_idx)


# 3.3 事件向量化
def event2vec(event_lists):
    sequences = []
    for company_events in event_lists:  # 每个temp是一个企业的当年违规事件列表
        events = []
        times = []
        counts = []
        for event in company_events:
            events.append(event[1])  # Use violation type as input feature
            times.append(event[2])  # Use violation time as input feature
            counts.append(event[3])
        sequences.append([events, times, counts])  # sequences是所有企业的违规事件列表
    # Find the maximum sequence length
    max_length = max([len(seq[0]) for seq in sequences])
    # Pad sequences to the maximum length
    padded_sequences = []
    for seq in sequences:
        padded_events = np.pad(seq[0], (0, max_length - len(seq[0]))).tolist()
        padded_times = np.pad(seq[1], (0, max_length - len(seq[1]))).tolist()
        padded_counts = np.pad(seq[2], (0, max_length - len(seq[2]))).tolist()
        padded_sequences.append([padded_events, padded_times, padded_counts])
    # Convert sequences to PyTorch tensors
    padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)  # company_num* 3组列表 *29位（每个列表的都是29位）
    return padded_sequences


train_event = event2vec(train_events)
valid_event = event2vec(valid_events)
test_event = event2vec(test_events)

'''
  04 RGAT
'''
from graph import Graph
from trainingdata import TrainingData

label_path = './01_rgcn_data/01_label_nt/' + year + '.nt'
file_path = './01_rgcn_data/02_triple_nt/' + year + '.nt'
graph = Graph()
graph.init_graph(file_path)
graph.init_graph_label(label_path)
graph.create_edge_data()
graph.print_graph_statistics()
training_data = TrainingData()
training_data.create_training_data(graph)
node_num = len(graph.enum_nodes.keys())
rel_num = (2 * len(graph.enum_relations.keys()) + 1)  # remember the inverse relations in the edge data
num_class = len(graph.enum_classes.keys())
edge_index = graph.edge_index
edge_type = graph.edge_type

'''
  05 基线Model,node2vec
'''
args = parser.parse_args()
device = torch.device("cpu")
set_random_seed(14)
nn.Softmax()
criterion = torch.nn.CrossEntropyLoss()
com_initial_emb = pd.read_pickle('./00_basic_data/01_meta/' + year + '.pkl')  # 导入meta 向量
# zeros_array = np.zeros((200,32))
# com_initial_emb = np.concatenatSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSe((com_initial_emb, zeros_array), axis=0)
# rsikgnn = RiskGNN(args.input_dim, args.output_dim, total_company_num, device, com_initial_emb, node_num, rel_num,
#                   num_class, audit_type, num_heads=1, dropout=0.2, norm=True)
rsikgnn = RiskGNN(args.input_dim, args.output_dim, total_company_num, device, com_initial_emb, node_num, rel_num,
                  num_class, num_heads=1, dropout=0.2, norm=True)
classifier = Classifier((args.output_dim)+3, 2).to(device)
model = nn.Sequential(rsikgnn, classifier)
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.008)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-6)

'''
  06 超图构建
'''

train_hyp_graph = {}
valid_hyp_graph = {}
test_hyp_graph = {}


# 0.1
def createhyper(line_key, temp_list, five_dict, line_com):
    if line_key not in temp_list:
        temp_list.append(line_key)
        five_dict[str(line_key)] = []
        five_dict[str(line_key)].append(line_com)
    else:
        five_dict[str(line_key)].append(line_com)


# 0.2
tb_index = pd.read_excel('./00_basic_data/00_comindex/new/' + year + '.xlsx')
tb_index = tb_index.fillna('')
index_dict = {}
for _, row in tb_index.iterrows():
    index_dict[str(row['代码'])] = row['id']


# 01
def stockdata():
    tb_stock1 = pd.read_excel('./01_hyper_data/01_stock/' + year + '.xlsx')
    tb_stock1 = tb_stock1.fillna('')
    stockPer_dict01 = []
    for _, row in tb_stock1.iterrows():
        if str(row['证券代码']) in index_dict:
            stockPer_dict01.append({'股东ID': row['股东ID'], '证券代码': index_dict[str(row['证券代码'])]})
    stock_list_train, stock_list_valid, stock_list_test = [], [], []
    stock_dict_train, stock_dict_valid, stock_dict_test = {}, {}, {}
    for line in tqdm(stockPer_dict01):
        if line['证券代码'] in train_idx:
            createhyper(line['股东ID'], stock_list_train, stock_dict_train, line['证券代码'])
        if line['证券代码'] in valid_idx:
            createhyper(line['股东ID'], stock_list_valid, stock_dict_valid, line['证券代码'])
        if line['证券代码'] in test_idx:
            createhyper(line['股东ID'], stock_list_test, stock_dict_test, line['证券代码'])
    train_hyp_graph['stock'] = stock_dict_train
    valid_hyp_graph['stock'] = stock_dict_valid
    test_hyp_graph['stock'] = stock_dict_test


# stockdata()

# 02
def managerdata():
    tb_manager1 = pd.read_excel('./01_hyper_data/02_manager/' + year + '.xlsx')
    tb_manager1 = tb_manager1.fillna('')
    managerPer_dict02 = []
    for _, row in tb_manager1.iterrows():
        if str(row['证券代码']) in index_dict:
            managerPer_dict02.append({'人员ID': row['人员ID'], '证券代码': index_dict[str(row['证券代码'])]})
    manager_list_train, manager_list_valid, manager_list_test = [], [], []
    manager_dict_train, manager_dict_valid, manager_dict_test = {}, {}, {}
    for line in tqdm(managerPer_dict02):
        if line['证券代码'] in train_idx:
            createhyper(line['人员ID'], manager_list_train, manager_dict_train, line['证券代码'])
        if line['证券代码'] in valid_idx:
            createhyper(line['人员ID'], manager_list_valid, manager_dict_valid, line['证券代码'])
        if line['证券代码'] in test_idx:
            createhyper(line['人员ID'], manager_list_test, manager_dict_test, line['证券代码'])
    train_hyp_graph['manager'] = manager_dict_train
    valid_hyp_graph['manager'] = manager_dict_valid
    test_hyp_graph['manager'] = manager_dict_test


# managerdata()

# 03
def auditdata():
    tb_audit1 = pd.read_excel('./01_hyper_data/03_audit/' + year + '.xlsx')
    tb_audit1 = tb_audit1.fillna('')
    auditPer_dict03 = []
    for _, row in tb_audit1.iterrows():
        if str(row['证券代码']) in index_dict:
            auditPer_dict03.append({'签字会计师ID': row['签字会计师ID'], '证券代码': index_dict[str(row['证券代码'])]})
    audit_list_train, audit_list_valid, audit_list_test = [], [], []
    audit_dict_train, audit_dict_valid, audit_dict_test = {}, {}, {}
    for line in tqdm(auditPer_dict03):
        if line['证券代码'] in train_idx:
            createhyper(line['签字会计师ID'], audit_list_train, audit_dict_train, line['证券代码'])
        if line['证券代码'] in valid_idx:
            createhyper(line['签字会计师ID'], audit_list_valid, audit_dict_valid, line['证券代码'])
        if line['证券代码'] in test_idx:
            createhyper(line['签字会计师ID'], audit_list_test, audit_dict_test, line['证券代码'])
    train_hyp_graph['audit'] = audit_dict_train
    valid_hyp_graph['audit'] = audit_dict_valid
    test_hyp_graph['audit'] = audit_dict_test


# auditdata()

# 04
def controldata():
    tb_control1 = pd.read_excel('./01_hyper_data/04_control/' + year + '.xlsx')
    tb_control1 = tb_control1.fillna('')
    controlPer_dict04 = []
    for _, row in tb_control1.iterrows():
        if str(row['证券代码']) in index_dict:
            controlPer_dict04.append({'实际控制人ID': row['实际控制人ID'], '证券代码': index_dict[str(row['证券代码'])]})
    control_list_train, control_list_valid, control_list_test = [], [], []
    control_dict_train, control_dict_valid, control_dict_test = {}, {}, {}
    for line in tqdm(controlPer_dict04):
        if line['证券代码'] in train_idx:
            createhyper(line['实际控制人ID'], control_list_train, control_dict_train, line['证券代码'])
        if line['证券代码'] in valid_idx:
            createhyper(line['实际控制人ID'], control_list_valid, control_dict_valid, line['证券代码'])
        if line['证券代码'] in test_idx:
            createhyper(line['实际控制人ID'], control_list_test, control_dict_test, line['证券代码'])
    train_hyp_graph['control'] = control_dict_train
    valid_hyp_graph['control'] = control_dict_valid
    test_hyp_graph['control'] = control_dict_test


# controldata()

# 05
def investdata():
    tb_invest1 = pd.read_excel('./01_hyper_data/05_invest/' + year + '.xlsx')
    tb_invest1 = tb_invest1.fillna('')
    investPer_dict05 = []
    for _, row in tb_invest1.iterrows():
        if str(row['证券代码']) in index_dict:
            investPer_dict05.append({'被投资方代码': row['被投资方代码'], '证券代码': index_dict[str(row['证券代码'])]})
    invest_list_train, invest_list_valid, invest_list_test = [], [], []
    invest_dict_train, invest_dict_valid, invest_dict_test = {}, {}, {}
    for line in tqdm(investPer_dict05):
        if line['证券代码'] in train_idx:
            createhyper(line['被投资方代码'], invest_list_train, invest_dict_train, line['证券代码'])
        if line['证券代码'] in valid_idx:
            createhyper(line['被投资方代码'], invest_list_valid, invest_dict_valid, line['证券代码'])
        if line['证券代码'] in test_idx:
            createhyper(line['被投资方代码'], invest_list_test, invest_dict_test, line['证券代码'])
    train_hyp_graph['invest'] = invest_dict_train
    valid_hyp_graph['invest'] = invest_dict_train
    test_hyp_graph['invest'] = invest_dict_test


# investdata()

# print("len", len(train_hyp_graph['invest']))  # 1000/87/1187/60/23

# train_hyp = []
# for i in ['stock', 'manager', 'control', 'invest']:
#     train_hyp += [gen_attribute_hg(total_company_num, train_hyp_graph[i], X=None)]
# valid_hyp = []
# for i in ['stock', 'manager', 'control', 'invest']:
#     valid_hyp += [gen_attribute_hg(total_company_num, valid_hyp_graph[i], X=None)]
# test_hyp = []
# for i in ['stock', 'manager', 'control', 'invest']:
#     test_hyp += [gen_attribute_hg(total_company_num, test_hyp_graph[i], X=None)]

'''
  07 训练与验证
'''
best_re = 0
best_f1 = 0
#Generative data can be added here
# with open('./03_generate/generate.pkl', 'rb') as f:
#     fake = pickle.load(f)
# train_label = np.append(train_label, np.ones(200))

gmean_vaild = []
gmean_train = []
acc_train = []
acc_vaild = []
acc_train.append(0)
acc_vaild.append(0)
# train_label = np.append(train_label, np.zeros(100))
for epoch in np.arange(args.n_epoch):
    st = time.time()
    # 01 训练
    model.train()
    train_losses = []
    # torch.cuda.empty_cache()
    # company_emb = rsikgnn.forward(train_risk_data, train_hyp, train_id, x_train, edge_index,edge_type)  # old版带Hyper
    company_emb = rsikgnn.forward(train_risk_data, train_id, edge_index, edge_type, train_event)  # 新版.训练集转化而成的企业向量，约3000条数据，张量维度为xxx，15

    # company_emb = torch.cat((company_emb, fake), dim=0)

    res = classifier.forward(company_emb)
    loss = criterion(res, torch.LongTensor(train_label))
    pred = res.argmax(dim=1)
    ac1 = acc(train_label, pred)
    pr1 = pre(train_label, pred)
    re1 = rec(train_label, pred)
    g_mean2 = gmean([pr1, re1])
    gmean_train.append(g_mean2)
    acc_train.append(ac1)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    train_losses += [loss.cpu().detach().tolist()]
    # train_step += 1
    scheduler.step()

    del res, loss

    # 02 验证
    model.eval()
    with torch.no_grad():
        # company_emb = rsikgnn.forward(valid_risk_data, valid_hyp, valid_id, x_valid, edge_index, edge_type) # old版带Hyper
        company_emb = rsikgnn.forward(valid_risk_data, valid_id, edge_index, edge_type, valid_event)  # 新版
        res = classifier.forward(company_emb)
        loss = criterion(res, torch.LongTensor(valid_label))
        pred = res.argmax(dim=1)
        ac = acc(valid_label, pred)
        pr = pre(valid_label, pred)
        re = rec(valid_label, pred)
        f = f1(valid_label, pred)
        rc = roc(valid_label, res[:, 1])
        g_mean = gmean([pr, re])
        if f > best_f1 and pr > 0.6 and re>best_re:
            best_re = re
            best_f1 = f
            torch.save(model, './model_save/%s.pkl' % (args.conv_name))
            print('UPDATE!!!')
        et = time.time()
        print((
                  "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid AUC: %.4f Valid G-mean: %.4f") % \
              (epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), ac, pr, re, f, rc,g_mean))
        del res, loss
        gmean_vaild.append(g_mean)
        acc_vaild.append(ac)
        if epoch + 1 == args.n_epoch:
            # company_emb = rsikgnn.forward(test_risk_data, test_hyp, test_id, x_test, edge_index, edge_type) # old版带Hyper
            company_emb = rsikgnn.forward(test_risk_data, test_id, edge_index, edge_type, test_event)  # 新版
            res = classifier.forward(company_emb)
            pred = res.argmax(dim=1)
            ac = acc(test_label, pred)
            pr = pre(test_label, pred)
            re = rec(test_label, pred)
            f = f1(test_label, pred)
            rc = roc(test_label, res[:, 1])


            print(
                'Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f Last Test AUC: %.4f' % (
                    ac, pr, re, f, rc))
            fig, ax = plt.subplots()
            # 绘制准确率曲线
            ax.plot(acc_train, label='Training ACC', color='blue', linestyle='-')
            # 绘制召回率曲线
            ax.plot(acc_vaild, label='Vaildation ACC', color='red', linestyle='-')
            # 添加图例
            ax.legend()
            # 添加标题和轴标签
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            # 设置纵轴的范围为0到1.0，确保包含0
            ax.set_ylim(0, 1)
            # 设置纵轴的刻度为0.1的倍数
            ax.xaxis.set_ticks(np.arange(0, 110,10))
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
            # 如果您想要y轴的刻度标签也显示小数点，可以设置如下：
            ax.tick_params(axis='y', which='both', length=0)
            # 显示网格
            # ax.grid(True)
            # 显示图形
            # plt.show()
'''
  08 测试
'''

def test(train_label=train_label):
    best_model = torch.load('./model_save/%s.pkl' % (args.conv_name))
    best_model.eval()
    rsikgnn, classifier = best_model
    with torch.no_grad():
        # company_emb = rsikgnn.forward(test_risk_data, test_hyp, test_id, x_test, edge_index, edge_type) # old版带Hyper
        company_emb = rsikgnn.forward(test_risk_data, test_id, edge_index, edge_type, test_event)  # old版带Hyper
        #The real vectors needed to generate the vectors
        # company_emb2 = rsikgnn.forward(train_risk_data, train_id, edge_index, edge_type, train_event)
        # with open('../../04_wgan-gp/data/2018train.pkl', 'wb') as f:
        #     pickle.dump(company_emb2, f)
        res = classifier.forward(company_emb)
        pred = res.argmax(dim=1)
        r2 = r2_score(test_label, pred)
        ac = acc(test_label, pred)
        pr = pre(test_label, pred)
        re = rec(test_label, pred)
        f = f1(test_label, pred)
        rc = roc(test_label, res[:, 1])
        g_mean = gmean([pr, re])
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(test_label)):
            true_label = test_label[i]
            pred_label = pred[i]

            # 计算tp，fp，tn，fn
            if true_label == 1 and pred_label == 1:
                tp += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 1 and pred_label == 0:
                fn += 1
            elif true_label == 0 and pred_label == 0:
                tn += 1

        # 打印结果
        print("TP:", tp)
        print("FP:", fp)
        print("TN:", tn)
        print("FN:", fn)
        print("R2:", r2)
        print(' Test Acc: %.4f Best Test Pre: %.4f  Test Recall: %.4f  Test F1: %.4f  Test AUC: %.4f Test G-mean: %.4f' % (
            ac, pr, re, f, rc,g_mean))
        print(sum(pred))


test()
