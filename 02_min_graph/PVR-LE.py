import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from py2neo import Graph, Node, Relationship, NodeMatcher
import math

url = 'bolt://127.0.0.1:7687'
user = 'neo4j'
password = '12345678'
# 创建图数据库链接
graph = Graph(url, auth=(user, password))

'''
00 选择年份
'''
year = '2018'
'''
01 建立networks  
'''
url = 'bolt://127.0.0.1:7687'
user = 'neo4j'
password = '12345678'
# 创建图数据库链接
graph = Graph(url, auth=(user, password))
mathcer = NodeMatcher(graph)
G = nx.Graph()


def createnetworks():
    cypherall = "MATCH (a)-[b]->(c) RETURN id(a),id(b),id(c),a.comid,c.comid"
    dfall = graph.run(cypherall).to_table()
    pairlist = []
    head_tail = []
    for line in tqdm(dfall):
        head = str(line[0])
        tail = str(line[2])
        head_tail.append(head + '_' + tail)
        if (tail + '_' + head) not in head_tail:
            pair = (int(line[0]), int(line[2]))
            pairlist.append(pair)
    G.add_edges_from(pairlist)  # 添加多条边


createnetworks()

'''
02  生成违规列表
'''
illegal_next = []  # 次年违规企业
illegal_this = []  # 今年违规企业
index_list = []  #


def this_next_label():
    # 数据处理 11 导入 违约企业标签
    # new为去掉非关系孤立企业后的数据
    tb_unique =pd.read_excel('../04_hgnn/hypergnn/00_basic_data/' + year + '.xlsx')
    for _, row in tb_unique.iterrows():
        this_comid = str(int(row['证券代码'])).zfill(6)  # 证券代码
        index_list.append(this_comid)
        if row['违规标签'] == 1:
            illegal_next.append(this_comid)  # 明年违法的comid
        if row['上年度违规'] == 1:
            illegal_this.append(this_comid)  # 今年违法的comid


this_next_label()

'''
04 Pagerank
'''
doublist_dict = []
com_part_dict = []


def dynamic_risk_allocation(x):
    if x <= 200:
        y = 1 / (0.03 + 1 * (x ** 0.5))
    else:
        y = 0.1
    return y


# 4.1 求PR
def PR(m, initial_pr):
    i = 0
    pr0 = initial_pr
    temp_pr = initial_pr
    aa = 0.7
    while 1:
        # a=0.7
        # b=0.3
        a = round((aa * dynamic_risk_allocation(1 + i)), 3)
        b = float(1 - a)
        Pr = a * np.dot(m, temp_pr) + b * pr0
        diff = np.abs(temp_pr - Pr)  # 计算两个数组对应位置上的绝对值差
        is_small_diff = np.all(diff < 0.001)  # 判断绝对值差是否都小于0.01
        if is_small_diff:
            break
        else:
            temp_pr = Pr
        i += 1
        # a=round((a/ math.log(int(i+2))),3)
        if i == 16:
            break
    # print('求pr值迭代%d次' % i, np.round(temp_pr, 3))
    return np.round(temp_pr, 3)


# 4.2 归一化
def normalization(m):
    column_sums = m.sum(axis=0)  # 计算每列元素之和
    normalized_matrix = np.where(column_sums == 0, m, m / column_sums[np.newaxis, :])
    return normalized_matrix


# 4.3  求illegal_id
def get_illegal_nodeid():
    illegal_id = []
    for line in illegal_this:
        cypherall1 = "MATCH (n:Company)  where n.comid='" + line + "' RETURN n.comid,id(n)"
        dfall = graph.run(cypherall1).to_table()
        if len(dfall) != 0:
            illegal_id.append(dfall[0][1])  # 今年违法的ID
    return illegal_id


# 4.3 传播
def riskrank():
    illegal_id = get_illegal_nodeid()
    comindexlist = []
    pr0 = []  # 初始风险列表
    # 矩阵的大小规模等于每个社区的数组长*长
    M = np.zeros((len(G.nodes), len(G.nodes)))
    row = 0
    # B、依次循环社区中的每个节点，一个节点的邻居数组--对应矩阵M中的一行
    for node_idx in list(G.nodes):
        # 00 求对应的comid,也可以查表
        cyphercom = "MATCH (a:Company)  where id(a)=" + str(node_idx) + " RETURN a.comid"
        dfcom = graph.run(cyphercom).to_table()
        if len(dfcom) != 0:
            comindexlist.append(str(dfcom[0][0]).zfill(6))
            # com_part_dict.append({'代码': str(dfcom[0][0]).zfill(6), 'nodevecpart': partid})
        # 01 初始风险值
        if node_idx not in illegal_id:
            pr0.append(float(0))
        else:
            pr0.append(float(1))
        # 02 求邻接权重
        cypher_ner = "MATCH (a:Company)-[b]-(c)  where id(a)=" + str(node_idx) + " RETURN id(c),b.percentage"
        ner_list = graph.run(cypher_ner).to_table()
        for ner in ner_list:  # 对于每个邻接阶段，确保邻接的节点也被划分到源节点的社区,ner[0]是邻接阶段id,ner[1]是对应的路径权重
            idx = list(G.nodes).index(ner[0])  # row随循环逐行递增，index求每行的指定列的权重
            M[row][idx] = ner[1]
        row += 1
    M_risk = normalization(M)  # 归一化
    Pr0 = np.array(pr0).reshape(len(pr0), 1)
    risklist = PR(M_risk, Pr0)
    # 每个group的index和risk
    doublist_dict.append({'comindexlist': comindexlist, 'risklist': risklist})
    # for ri in risklist:
    #         print(ri)
    # print("第",id, "社区___________________________________")


riskrank()

'''
05  计算邻接节点
'''
# 计算每个节点的度和名称
cypher_degree = "MATCH (n) RETURN id(n) AS node_id, size((n)-[]-()) AS degree, n.comid AS com_id"
result = graph.run(cypher_degree)
data_dict = {}
for row in result:
    node_id = row["node_id"]
    degree = row["degree"]
    comidx = row["com_id"]
    # print(f"Node {node_id} 的度为 {degree}，comid为 {comidx}")
    data_dict[comidx] = {"node_id": node_id, "degree": degree}


# 查询函数
def query_data_by_name(comidxx):
    if comidxx in data_dict:
        return data_dict[comidxx]
    else:
        return None


'''
06 衰减函数
'''


def calculate_dependent_variable(x):
    ww = 2
    bb = 0.11
    min_threshold=0.2
    y = 1 / (1 + ww * (x ** bb))
    if y<min_threshold:
        y = min_threshold
    return y


'''
07 输出
'''


def outputrisk():
    risk_result_dict = []
    for line in tqdm(doublist_dict):
        indexid = 0
        while indexid < len(line['comindexlist']):
            if line['comindexlist'][indexid] in illegal_next:
                label = 1
            else:
                label = 0
            temp_find = line['comindexlist'][indexid]
            # 只保留待预测企业的风险信息，其余无关邻接风险剔除
            if temp_find in index_list:
                data = query_data_by_name(temp_find)
                degree = data["degree"]
                decay_weight = calculate_dependent_variable(degree)
                # [0]是由于每个risk都被装入了list,[],[],[]
                risk_result_dict.append({"证券代码": line['comindexlist'][indexid],
                                         "风险值": round(float((line['risklist'][indexid][0]) * decay_weight) * 10, 3),
                                         "违规标签": label})
                # risk_result_dict.append({"证券代码": line['comindexlist'][indexid],
                #                          "风险值": round(float((line['risklist'][indexid][0]) / degree) * 10, 3),
                #                          "违规标签": label})
                # risk_result_dict.append({"证券代码": line['comindexlist'][indexid],"风险值": line['risklist'][indexid][0],"违规标签": label})
                indexid += 1
            else:
                indexid += 1
    df = pd.DataFrame(risk_result_dict, columns=['证券代码', '风险值', '违规标签'])
    print(df.corr())
    # type="audit_on"
    df.to_excel('./data/06_risk/'+ year +'.xlsx', index=False)


outputrisk()
