# 读取表格
import pandas as pd
from tqdm import tqdm
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import requests, json
import pypinyin

def trans(str):
    header = {
        "User - Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
        "Host": "fanyi.youdao.com",
        "Referer": "https://fanyi.youdao.com/",
    }

    data = {
        'type': "AUTO",
        'i': str,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    res_data = requests.post('http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule', data=data,
                             headers=header)
    # 需要字符串转化为字典
    result = json.loads(res_data.text)
    # 切片指定字符：
    result1 = result["translateResult"][0][0]["tgt"]
    eng = ' '.join(result1.split()[:5])
    return eng


def pinyin(str):
    result1 = pypinyin.pinyin(str, style=pypinyin.NORMAL)
    result_ = [i[0] for i in result1]
    result3 = result_[0].capitalize() + ' ' + ''.join(result_[1:]).capitalize()
    return result3


url = 'bolt://127.0.0.1:7687'
user = 'neo4j'
password = '12345678'
# 创建图数据库链接
graph = Graph(url, auth=(user, password))
mathcer = NodeMatcher(graph)
#年份
thisyear="2018"

# 0 创建企业节点（名称+ID）
def createNameNode():
    num_com_node = 0
    tb_name0 = pd.read_excel('../01_data/0_comindex/'+thisyear+'.xlsx')
    name_dict00 = []
    for _, row in tb_name0.iterrows():
        stockId1 = str(row['证券代码']).zfill(6)  # 证券代码
        name_dict00.append({'证券代码': stockId1, '公司简称': row['公司简称']})
    for com in tqdm(name_dict00):
        eng = com['公司简称']
        company_node01 = Node('Company', ename=eng, cname=com['公司简称'], comid=com['证券代码'])
        graph.create(company_node01)
        num_com_node += 1
    print("已创建", num_com_node, "个企业节点")


def remedynode(id):  # 弥补节点
    company_node01 = Node('Company', ename=id, cname=id, comid=id)  # 创建01企业节点,2个属性
    graph.create(company_node01)
    return company_node01


# 1 创建股东节点（名称+ID）
def createStoPerNodeAndRel():
    num_stockPer_node = 0
    num_stockPer_line = 0
    tb_stock1 = pd.read_excel('../01_data/a_stock/'+thisyear+'.xlsx')
    tb_stock1 = tb_stock1.fillna('')
    stockPer_dict01 = []
    for _, row in tb_stock1.iterrows():
        stockId2 = str(row['证券代码']).zfill(6)  # 证券代码
        stockPer_dict01.append({'证券代码': stockId2, '股东ID': row['股东ID'], '股东名称': row['股东名称'], '持股比例': row['持股比例']})
    for line in tqdm(stockPer_dict01):
        temp_node = mathcer.match('StockPer', stockid=str(line['股东ID'])).first()
        if temp_node == None:  # 以前未建立的
            # if len(line['股东名称'])>4:
            #     eng_stock = trans(line['股东名称'])
            # else:
            eng_stock = pinyin(line['股东名称'])
            stockPer_node02 = Node('StockPer', name=line['股东名称'], stockid=str(line['股东ID']),
                                   eng=str(eng_stock))  # 创建02股东节点,2个属性
            graph.create(stockPer_node02)
            num_stockPer_node += 1
            com_node = mathcer.match('Company', comid=line['证券代码']).first()
            if com_node != None:
                com_stockPer01 = Relationship(stockPer_node02, '持股', com_node, share=float(line['持股比例']))
                graph.create(com_stockPer01)
                num_stockPer_line += 1
                continue
            else:
                com_node = remedynode(line['证券代码'])
                com_stockPer01 = Relationship(stockPer_node02, '持股', com_node, share=float(line['持股比例']))
                graph.create(com_stockPer01)
                num_stockPer_line += 1
                continue
        else:
            com_node = mathcer.match('Company', comid=line['证券代码']).first()
            if com_node != None:
                com_stockPer01 = Relationship(temp_node, '持股', com_node, share=float(line['持股比例']))
                graph.create(com_stockPer01)
                num_stockPer_line += 1
                continue
            else:
                com_node = remedynode(line['证券代码'])
                com_stockPer01 = Relationship(temp_node, '持股', com_node, share=float(line['持股比例']))
                graph.create(com_stockPer01)
                num_stockPer_line += 1
    print("已创建", num_stockPer_node, "个股东节点")
    print("已创建", num_stockPer_line, "条持股关系")


# 2 创建董事长、董事、总经理、高层经理、高管节点（名称+ID）
def createManagerNodeAndRel():
    tb_manager2 = pd.read_excel('../01_data/b_executive/'+ thisyear +'.xlsx')
    tb_manager2 = tb_manager2.fillna('')
    managerPer_dict02 = []
    for _, row in tb_manager2.iterrows():
        stockId3 = str(row['证券代码']).zfill(6)  # 证券代码
        managerPer_dict02.append({'证券代码': stockId3, '人员ID': row['人员ID'], '姓名': row['姓名'], '具体职务': row['具体职务']})
    tempid1 = []
    for line in tqdm(managerPer_dict02):
        com_node = mathcer.match('Company', comid=line['证券代码']).first()
        if com_node == None:
            com_node = remedynode(line['证券代码'])
        # 开始导入1类节点
        if line['人员ID'] != "":
            line['人员ID'] = int(line['人员ID'])
            if line['人员ID'] not in tempid1:
                eng_per = pinyin(line['姓名'])
                leader_node03 = Node('Executive', ename=eng_per, cname=line['姓名'],
                                     leaderid=str(line['人员ID']))  # 创建03董事长节点,2个属性
                graph.create(leader_node03)
                tempid1.append(line['人员ID'])
                com_leader02 = Relationship(leader_node03, '任职', com_node, pram='executive')
                graph.create(com_leader02)
            else:
                find01 = mathcer.match('Executive', leaderid=str(line['人员ID'])).first()
                com_leader02 = Relationship(find01, '任职', com_node, pram='executive')
                graph.create(com_leader02)


# 4 控制人节点
def createControllerNodeAndRel():
    tb_control4 = pd.read_excel('../01_data/c_control/'+thisyear+'.xlsx')
    tb_control4 = tb_control4.fillna('')
    auditPer_dict04 = []
    for _, row in tb_control4.iterrows():
        stockId5 = str(row['证券代码']).zfill(6)  # 证券代码
        auditPer_dict04.append(
            {'证券代码': stockId5, '实际控制人ID': row['实际控制人ID'], '控制人名称': row['控制人名称'], '控股比例': row['控股比例']})
    tempid41 = []
    for line in tqdm(auditPer_dict04):
        com_node = mathcer.match('Company', comid=line['证券代码']).first()
        if com_node == None:
            com_node = remedynode(line['证券代码'])
        # 041 实际控制人
        if line['实际控制人ID'] != "":
            # line['实际控制人ID'] = int(line['实际控制人ID'])
            if line['实际控制人ID'] not in tempid41:
                eng_per = pinyin(line['控制人名称'])
                controller_node05 = Node('Controller',  ename=eng_per, cname=line['控制人名称'],controllerid=str(line['实际控制人ID']))  # 创建031 机构节点,2个属性
                graph.create(controller_node05)
                tempid41.append(line['实际控制人ID'])
                com_controller04 = Relationship(controller_node05, '控股', com_node, weight=float(line['控股比例']))
                graph.create(com_controller04)
            else:
                temp_node041 = mathcer.match('Controller', controllerid=str(line['实际控制人ID'])).first()
                com_controller04 = Relationship(temp_node041, '控股', com_node, weight=float(line['控股比例']))
                graph.create(com_controller04)


# 05 投资
def createInvestNodeAndRel():
    tb_invest5 = pd.read_excel('../01_data/d_invest/'+thisyear+'.xlsx')
    tb_invest5 = tb_invest5.fillna('')
    investPer_dict05 = []
    for _, row in tb_invest5.iterrows():
        stockId61 = str(row['证券代码']).zfill(6)  # 证券代码
        stockId62 = str(row['被投资方代码']).zfill(6)  # 证券代码
        investPer_dict05.append(
            {'证券代码': stockId61, '被投资方代码': stockId62,'被投资方名称': row['被投资方名称'], '投资金额': row['投资金额']})
    for line in tqdm(investPer_dict05):
        com_node = mathcer.match('Company', comid=line['证券代码']).first()
        if com_node == None:
            com_node = remedynode(line['证券代码'])
        if line['被投资方代码'] != "":
            invest_node = mathcer.match('Company', comid=line['被投资方代码']).first()
            if invest_node == None:
                invest_node = remedynode(line['被投资方代码'])
            # 051 被投资方
            com_invest01 = Relationship(com_node, '投资', invest_node, investcount=float(line['投资金额']))
            graph.create(com_invest01)


# 06_legaldata 违法
def createLegalNodeAndRel():
    # tb_invest6 = pd.read_excel('../01_dataprocess/f_violate/00_event/'+thisyear+'.xlsx')
    tb_invest6 = pd.read_excel('../01_data/e_violate/' + thisyear + '.xlsx')
    tb_invest6 = tb_invest6.fillna('')
    investPer_dict06 = []
    for _, row in tb_invest6.iterrows():
        stockId61 = str(row['证券代码']).zfill(6)  # 证券代码
        investPer_dict06.append({'证券代码': stockId61, '事件ID': row['事件ID']})
    for event in tqdm(investPer_dict06):
        event_node061 = Node('Violation', eventid=str(event['事件ID']))  # 创建01企业节点,2个属性
        graph.create(event_node061)
        com_node = mathcer.match('Company', comid=event['证券代码']).first()
        if com_node == None:
            com_node = remedynode(event['证券代码'])
        com_violate = Relationship(com_node, '违规', event_node061, pram="violate")
        graph.create(com_violate)


createNameNode()  # 00
createStoPerNodeAndRel()  # 01
createManagerNodeAndRel()  # 02
createControllerNodeAndRel()  # 03
createInvestNodeAndRel()  # 04
createLegalNodeAndRel() # 05
# X 删除
# MATCH (n:Company) detach delete n
# MATCH (n:StockPer) detach delete n
# MATCH (n:Executive) detach delete n
# MATCH (n:AuditorPer) detach delete n
# MATCH (n:Controller) detach delete n
# MATCH (a:Company)-[rel:投资]-(b:Company ) DELETE rel
# MATCH (n:Violation) detach delete n

#MATCH (a:CityTest {name:'青岛'}) DELETE a
# match (n) detach delete n
# MATCH (a:Company)-[:持股]->(b:StockPer {stockid:5013840}) RETURN a,b
# MATCH (n:Company) RETURN n.comid, id(n)