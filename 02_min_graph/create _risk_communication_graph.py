import pandas as pd
from tqdm import tqdm
from py2neo import Graph, Node, Relationship, NodeMatcher

url = 'bolt://127.0.0.1:7687'
user = 'neo4j'
password = '12345678'
# 创建图数据库链接
graph = Graph(url, auth=(user, password))
mathcer = NodeMatcher(graph)

def cteatemin():
    year="2018"
    # 先查违规表
    tb_vio =  pd.read_excel('../04_hgnn/hypergnn/00_basic_data/' + year + '.xlsx')
    tb_vio['证券代码'] = tb_vio['证券代码'].astype(str).str.zfill(6)
    # 构建图谱
    com_dictlist = []
    id_tmep=[]
    # tb_com = pd.read_excel('./05_final_weight/' + year + '/' + year + '.xlsx')

    tb_com = pd.read_excel('./data/05_final_weight/' + year + '/' + year + 'dropaudit.xlsx')
    for _, row in tqdm(tb_com.iterrows()):
        #头节点
        head = str(int(row['企业1'])).zfill(6)  # 证券代码
        result_1 = tb_vio[tb_vio['证券代码'] == head]
        if result_1.empty:
            zero_1 = 0
            count_1 = 0
        else:
            zero_1 = result_1.iloc[0]['上年度违规']
            count_1 = result_1.iloc[0]['上年度违规次数']
        #尾节点
        tail = str(int(row['企业2'])).zfill(6)  # 证券代码
        result_2 = tb_vio[tb_vio['证券代码'] == tail]
        if result_2.empty:
            zero_2 = 0
            count_2 = 0
        else:
            zero_2 = result_2.iloc[0]['上年度违规']
            count_2 = result_2.iloc[0]['上年度违规次数']
        #头尾概率
        weight= round( row['联合概率'], 4)
        com_dict={'head': head, 'tail': tail, 'score':weight,'zero_1':zero_1,'count_1':count_1,'zero_2':zero_2,'count_2':count_2}
        com_dictlist.append(com_dict)
    for com in tqdm(com_dictlist):
        if com['head'] not in id_tmep:
            company_node = Node('Company', name=com['head'], comid=com['head'],zero=str(com['zero_1']),viocount=str(com['count_1']))  # 创建01企业节点,2个属性
            graph.create(company_node)
            id_tmep.append(com['head'])
        if com['tail'] not in id_tmep:
            company_node = Node('Company', name=com['tail'], comid=com['tail'],zero=str(com['zero_2']),viocount=str(com['count_2']))  # 创建01企业节点,2个属性
            graph.create(company_node)
            id_tmep.append(com['tail'])
        head_node = mathcer.match('Company', comid=com['head']).first()
        tail_node = mathcer.match('Company', comid=com['tail']).first()
        com_com = Relationship(head_node, str(com['score']), tail_node, percentage=float(com['score']))
        graph.create(com_com)
cteatemin()