import os

from tqdm import tqdm
from py2neo import Graph
import pandas as pd

def connect_to_neo4j():
    url = 'bolt://127.0.0.1:7687'
    user = 'neo4j'
    password = '12345678'
    return Graph(url, auth=(user, password))

def process_holdings_data(graph, year):
    cypher = f"MATCH (a:Company)-[b:持股]-(c)-[d:持股]-(e:Company) RETURN a.comid,b.share,d.share,e.comid"
    df = graph.run(cypher).to_table()
    df = pd.DataFrame(df)
    new_column_names = {0: '企业1', 1: '参数1', 2: '参数2', 3: '企业2'}
    df = df.rename(columns=new_column_names)
    df = df.sort_values(['参数1', '参数2'], ascending=[False, False])
    s_df = df.drop_duplicates(subset=['企业1', '企业2'], keep='first')
    unique_combinations = set()
    rows_to_keep = []
    for index, row in s_df.iterrows():
        head_entity = row['企业1']
        tail_entity = row['企业2']
        shuncheng = str(head_entity) + str(tail_entity)
        re = str(tail_entity) + str(head_entity)
        if re not in unique_combinations:
            unique_combinations.add(shuncheng)
            rows_to_keep.append(index)
    new_df = df.iloc[rows_to_keep]
    new_df = new_df.reset_index(drop=True)
    new_df.to_excel(f'./data/00_rawdata/{year}/持股.xlsx', index=False)

def process_positions_data(graph, year):
    cypher = f"MATCH (a:Company)-[b:任职]-(c)-[d:任职]-(e:Company) RETURN a.comid,c.leaderid,e.comid"
    df = graph.run(cypher).to_table()
    df = pd.DataFrame(df)
    new_column_names = {0: '企业1', 1: '人员id', 2: '企业2'}
    df = df.rename(columns=new_column_names)
    unique_combinations = set()
    rows_to_keep = []
    for index, row in df.iterrows():
        head_entity = row['企业1']
        tail_entity = row['企业2']
        shuncheng = str(head_entity) + str(tail_entity)
        re = str(tail_entity) + str(head_entity)
        if re not in unique_combinations:
            unique_combinations.add(shuncheng)
            rows_to_keep.append(index)
    new_df = df.iloc[rows_to_keep]
    new_df = new_df.reset_index(drop=True)
    new_df.to_excel(f'./data/00_rawdata/{year}/任职.xlsx', index=False)
def process_control_data(graph, year):
    cypher = "MATCH (a:Company)-[b:控股]-(c)-[d:控股]-(e:Company) RETURN a.comid,b.weight,d.weight,e.comid"
    df = graph.run(cypher).to_table()
    df = pd.DataFrame(df)
    new_column_names = {0: '企业1', 1: '参数1', 2: '参数2', 3: '企业2'}
    df = df.rename(columns=new_column_names)
    df = df.sort_values(['参数1', '参数2'], ascending=[False, False])  # 按比例1降序，比例2降序排序
    s_df = df.drop_duplicates(subset=['企业1', '企业2'], keep='first')  # 保留重复组合中的第一行
    # ----去重
    # 创建一个集合用于存储删除重复行后的组合
    unique_combinations = set()
    rows_to_keep = []
    # 遍历每一行数据
    for index, row in s_df.iterrows():
        # 获取当前行的头实体和尾实体
        head_entity = row['企业1']
        tail_entity = row['企业2']
        shuncheng = str(head_entity) + str(tail_entity)
        # 交换头实体和尾实体的顺序
        re = str(tail_entity) + str(head_entity)
        # 检查该组合是否已经出现过（在集合中）
        if re not in unique_combinations:
            unique_combinations.add(shuncheng)
            rows_to_keep.append(index)
    # 根据保留行的索引，筛选出新的dataframe
    new_df = df.iloc[rows_to_keep]
    # 重置索引
    new_df = new_df.reset_index(drop=True)
    new_df.to_excel(f'./data/00_rawdata/{year}/控股.xlsx', index=False)
def process_invest_data(graph, year):
    cypher = "MATCH (a:Company)-[b:投资]-(c)-[d:投资]-(e:Company) RETURN a.comid,b.investcount,d.investcount,e.comid"
    # 执行下面这句 run(“语句”)就可以，返回table格式
    df = graph.run(cypher).to_table()
    df = pd.DataFrame(df)
    new_column_names = {0: '企业1', 1: '参数1', 2: '参数2', 3: '企业2'}
    df = df.rename(columns=new_column_names)
    df = df.groupby(['企业1', '企业2'], as_index=False).sum()
    df = df.drop_duplicates(subset=['企业1', '企业2'])
    # ----去重
    # 创建一个集合用于存储删除重复行后的组合
    unique_combinations = set()
    rows_to_keep = []
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 获取当前行的头实体和尾实体
        head_entity = row['企业1']
        tail_entity = row['企业2']
        shuncheng = str(head_entity) + str(tail_entity)
        # 交换头实体和尾实体的顺序
        re = str(tail_entity) + str(head_entity)
        # 检查该组合是否已经出现过（在集合中）
        if re not in unique_combinations:
            unique_combinations.add(shuncheng)
            rows_to_keep.append(index)

    # 根据保留行的索引，筛选出新的dataframe
    new_df = df.iloc[rows_to_keep]
    # 重置索引
    new_df = new_df.reset_index(drop=True)
    new_df.to_excel(f'./data/00_rawdata/{year}/投资.xlsx', index=False)
    print("End of phase 1")
def sum_amount(year):
    file_types = ['控股', '持股', '投资']
    for file_type in file_types:
        file_name = f'./data/00_rawdata/{year}/{file_type}.xlsx'
        if not os.path.exists(file_name):
            continue
        df = pd.read_excel(file_name)
        df['企业1'] = df['企业1'].astype(str).str.zfill(6)
        df['企业2'] = df['企业2'].astype(str).str.zfill(6)
        df['参数之和'] = df['参数1'] + df['参数2']
        new_file_name = f'./data/00_rawdata/{year}/new/{file_type}new.xlsx'
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
        df.to_excel(new_file_name, index=False)

#01 持股
def chigu(year):
    # for year in tqdm(range(2017, 2022)):
    df = pd.read_excel('./data/00_rawdata/'+str(year)+'/new/持股new.xlsx')
    newdf = df[(df['参数1'] >= 2.9) & (df['参数2'] >= 2.9)]
    newdf.drop(['参数1', '参数2'], axis=1, inplace=True)
    newdf.loc[:, '企业1'] = newdf['企业1'].astype(str).str.zfill(6)
    newdf.loc[:, '企业2'] = newdf['企业2'].astype(str).str.zfill(6)
    newdf.to_excel('./data/01_sixrel/'+str(year)+'/持股.xlsx', index=False)



#02 任职
def renzhi(year):
    # for year in tqdm(range(2017, 2021)):
    newdf = pd.read_excel('./data/00_rawdata/'+str(year)+'/任职.xlsx')
    # newdf.drop(['人员id'], axis=1, inplace=True)
    newdf['参数之和'] = '任职关系'
    newdf.loc[:, '企业1'] = newdf['企业1'].astype(str).str.zfill(6)
    newdf.loc[:, '企业2'] = newdf['企业2'].astype(str).str.zfill(6)
    newdf.to_excel('./data/01_sixrel/'+str(year)+'/任职.xlsx', index=False)
# renzhi()


#04 控股
def konggu(year):
    # for year in tqdm(range(2016, 2021)):
    df = pd.read_excel('./data/00_rawdata/'+str(year)+'/new/控股new.xlsx')
    newdf = df[(df['参数1'] >= 1.5) & (df['参数2'] >=1.5)]
    newdf.drop(['参数1', '参数2'], axis=1, inplace=True)
    newdf.loc[:, '企业1'] = newdf['企业1'].astype(str).str.zfill(6)
    newdf.loc[:, '企业2'] = newdf['企业2'].astype(str).str.zfill(6)
    newdf.to_excel('./data/01_sixrel/'+str(year)+'/控股.xlsx', index=False)
# konggu()

#05 二阶投资
def touzi_2(year):
    # for year in tqdm(range(2016, 2021)):
    df = pd.read_excel('./data/00_rawdata/'+str(year)+'/new/投资new.xlsx')
    newdf = df[(df['参数1'] >= 7000000) & (df['参数2'] >= 7000000)]
    newdf.drop(['参数1', '参数2'], axis=1, inplace=True)
    newdf.loc[:, '企业1'] = newdf['企业1'].astype(str).str.zfill(6)
    newdf.loc[:, '企业2'] = newdf['企业2'].astype(str).str.zfill(6)
    newdf.to_excel('./data/01_sixrel/'+str(year)+'/二阶投资.xlsx', index=False)
# touzi_2()

#06 一阶投资 数据直接拷贝 02_knowgraph/e_invest
def touzi_1(year):
    # for year in tqdm(range(2016, 2021)):
    df = pd.read_excel('../01_knowgraph/01_data/d_invest/deal_data/'+str(year)+'new.xlsx')
    newdf = df[(df['参数之和'] >= 400000)]
    newdf.loc[:, '企业1'] = newdf['企业1'].astype(str).str.zfill(6)
    newdf.loc[:, '企业2'] = newdf['企业2'].astype(str).str.zfill(6)
    newdf.to_excel('./data/01_sixrel/'+str(year)+'/一阶投资.xlsx', index=False)
    print("End of phase 2")

def unique(year):
    file_types = ['控股', '持股', '一阶投资', '二阶投资', '任职']
    for type_str in file_types:
        df_duplicate = pd.read_excel(f'./data/01_sixrel/{year}/{type_str}.xlsx')
        df_duplicate['企业1'] = df_duplicate['企业1'].astype(str).str.zfill(6)
        df_duplicate['企业2'] = df_duplicate['企业2'].astype(str).str.zfill(6)
        df_duplicate['排序列'] = df_duplicate[['企业1', '企业2']].apply(lambda x: tuple(sorted(x)), axis=1)
        df_unique = df_duplicate.drop_duplicates(subset=['排序列', '参数之和'])
        df_unique.drop('排序列', axis=1, inplace=True)
        df_unique.drop_duplicates(subset=['企业1', '企业2', '参数之和'], keep=False, inplace=True)
        df_unique = df_unique.drop(df_duplicate[df_duplicate['企业1'] == df_duplicate['企业2']].index)
        save_dir = f'./data/01_uniquerel/{year}/'
        os.makedirs(save_dir, exist_ok=True)
        df_unique.to_excel(f'{save_dir}{type_str}.xlsx', index=False)
        testa = set()
        duplicate_found = False
        for index, row in df_unique.iterrows():
            head_entity = row['企业1']
            tail_entity = row['企业2']
            shuncheng = str(head_entity) + str(tail_entity)
            re = str(tail_entity) + str(head_entity)
            if shuncheng in testa or re in testa:
                duplicate_found = True
                print(f"相反元组：{shuncheng} 顺承元组：{re}")
            testa.add(shuncheng)
        if not duplicate_found:
            print("无重复")

# 01 添加企业1与企业2的标签
def tag_label(year):
    file_types = ['控股', '持股', '一阶投资', '二阶投资', '任职']
    df2 = pd.read_excel('../04_hgnn/hypergnn/00_basic_data/' + year + '.xlsx')
    for type_str in file_types:
        df1 = pd.read_excel(f'./data/01_uniquerel/{year}/{type_str}.xlsx')
        df1['企业1上年度违规'] = 0
        df1['企业2上年度违规'] = 0
        df1['企业1违规标签'] = 0
        df1['企业2违规标签'] = 0
        for index, row in df1.iterrows():
            match_row1 = df2[df2['证券代码'] == row['企业1']]
            match_row2 = df2[df2['证券代码'] == row['企业2']]
            if not match_row1.empty:
                df1.at[index, '企业1上年度违规'] = match_row1['上年度违规'].values[0]
                df1.at[index, '企业1违规标签'] = match_row1['违规标签'].values[0]
            if not match_row2.empty:
                df1.at[index, '企业2上年度违规'] = match_row2['上年度违规'].values[0]
                df1.at[index, '企业2违规标签'] = match_row2['违规标签'].values[0]
        df1.loc[:, '企业1'] = df1['企业1'].astype(str).str.zfill(6)
        df1.loc[:, '企业2'] = df1['企业2'].astype(str).str.zfill(6)
        save_dir = f'./data/02_sixrel_weight/{year}/'
        os.makedirs(save_dir, exist_ok=True)
        df1.to_excel(f'{save_dir}{type_str}.xlsx', index=False)

# 02 计算每年的各类权重比
def weight(year):
    file_types = ['控股', '持股', '一阶投资', '二阶投资', '任职']
    for type_str in file_types:
        data = pd.read_excel(f'./data/02_sixrel_weight/{year}/{type_str}.xlsx')
        condition = ((data['企业1上年度违规'] == 1) & (data['企业2违规标签'] == 1)) | \
                    ((data['企业2上年度违规'] == 1) & (data['企业1违规标签'] == 1))
        row_count = len(data[condition])
        risk_weight = round(row_count / len(data), 4)
        df_weight = data.drop(columns=['参数之和', '企业1上年度违规', '企业2上年度违规', '企业1违规标签', '企业2违规标签'])
        df_weight['风险权重'] = risk_weight
        df_weight.loc[:, '企业1'] = df_weight['企业1'].astype(str).str.zfill(6)
        df_weight.loc[:, '企业2'] = df_weight['企业2'].astype(str).str.zfill(6)
        save_dir = f'./data/03_risk_weight/{year}/'
        os.makedirs(save_dir, exist_ok=True)
        df_weight.to_excel(f'{save_dir}{type_str}.xlsx', index=False)

def concat_excel_files(year):
    file_types = ['控股', '持股', '一阶投资', '二阶投资', '任职']
    columns_of_interest = ['企业1', '企业2', '风险权重']
    all_data = pd.DataFrame()

    for type_str in file_types:
        file_path = f'./data/03_risk_weight/{year}/{type_str}.xlsx'
        if os.path.exists(file_path):
            data = pd.read_excel(file_path, usecols=columns_of_interest)
            data['企业1'] = data['企业1'].astype(str).str.zfill(6)
            data['企业2'] = data['企业2'].astype(str).str.zfill(6)
            all_data = pd.concat([all_data, data], ignore_index=True)

    output_file = f'./data/04_all_weight/{year}new.xlsx'
    all_data.to_excel(output_file, index=False)


def complex_weight(year):
    df = pd.read_excel('./data/04_all_weight/' + year + 'new.xlsx')
    df['企业1'] = df['企业1'].astype(str).str.zfill(6)
    df['企业2'] = df['企业2'].astype(str).str.zfill(6)
    # 创建字典保存每个企业对应的概率累积结果
    probabilities = {}
    # 计算联合补集概率的乘积
    for index, row in df.iterrows():
        company1 = str(row['企业1']).strip()
        company2 = str(row['企业2']).strip()
        probability = float(row['风险权重'])
        key = (company1, company2) if company1 < company2 else (company2, company1)
        complement_probability = 1 - probability
        if key in probabilities:
            probabilities[key] *= complement_probability
        else:
            probabilities[key] = complement_probability
    # 创建结果DataFrame
    result_df = pd.DataFrame(columns=['企业1', '企业2', '联合概率'])
    # 填充结果DataFrame
    for (company1, company2), complement_probability in probabilities.items():
        joint_probability = 1 - complement_probability
        result_df = result_df.append({'企业1': company1, '企业2': company2, '联合概率': joint_probability},
                                     ignore_index=True)
    # 按照联合概率从高到低排序
    result_df = result_df.sort_values(by='联合概率', ascending=False)
    # 将结果保存为Excel表格
    # result_df.to_excel('./05_final_weight/' + year + '/' + year + '.xlsx', index=False)
    result_df.to_excel('./data/05_final_weight/' + year + '/' + year + 'dropaudit.xlsx', index=False)
    print("End of phase 3")

def main():
    graph = connect_to_neo4j()
    year = "2018"

    process_holdings_data(graph, year)
    process_positions_data(graph, year)
    process_control_data(graph, year)
    process_invest_data(graph, year)

    sum_amount(year)
    chigu(year)
    renzhi(year)
    konggu(year)
    touzi_2(year)
    touzi_1(year)

    unique(year)
    tag_label(year)
    weight(year)
    concat_excel_files(year)
    complex_weight(year)



if __name__ == "__main__":
    main()