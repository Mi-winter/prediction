import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from datetime import datetime
from dataclasses import dataclass
from typing import List
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 数据读取
@dataclass
class Location:
    lot_no: str
    address: str
    user: str
    area_sqm: str

@dataclass
class Disposal:
    date: str
    disposal_type: str
    premium: str
    purchaser: str
    location: List[Location]

# 读取JSON文件
def read_json_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print("JSON解析错误:", e)
    return data

# 将JSON数据转换为Disposal对象列表
def json_to_disposal_objects(json_data):
    disposals = []
    for item in json_data:
        locations = [Location(**loc) for loc in item['location']]
        disposal = Disposal(
            date=item['date'],
            disposal_type=item['disposal_type'],
            premium=item['premium'],
            purchaser=item['purchaser'],
            location=locations
        )
        disposals.append(disposal)
    return disposals

# 读取JSON文件并转换为Disposal对象列表
def read_and_convert_json(file_path):
    json_data = read_json_file(file_path)
    disposals = json_to_disposal_objects(json_data)
    return disposals

def visualization(disposal_df):
    # 绘制散点图，展示面积和溢价之间的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='area_sqm', y='premium', hue='disposal_type', data=disposal_df)
    plt.title('Relationship between Area and Premium')
    plt.xlabel('Area (sqm)')
    plt.ylabel('Premium')
    plt.grid(True)

    # 绘制箱线图，展示不同处置类型的面积分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='disposal_type', y='area_sqm', data=disposal_df)
    plt.title('Distribution of Area by Disposal Type')
    plt.xlabel('Disposal Type')
    plt.ylabel('Area (sqm)')
    plt.grid(True)

    # 绘制折线图，展示不同日期下的溢价情况
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='premium', hue='disposal_type', data=disposal_df)
    plt.title('Premium Over Dates')
    plt.xlabel('Date')
    plt.ylabel('Premium')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 绘制柱状图，展示不同土地用途的数量
    plt.figure(figsize=(10, 6))
    sns.countplot(x='user', data=disposal_df)
    plt.title('Count of Different Land Uses')
    plt.xlabel('Land Use')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def is_valid_date(date_str):
    # 正则表达式判断日期格式是否正确
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return re.match(pattern, date_str) is not None

def is_numeric(value):
    # 判断字符串是否为数字
    try:
        float(value)
        return True
    except ValueError:
        return False

def data_clean(disposals):
    filtered_disposals = []
    
    for disposal in disposals:
        if is_valid_date(disposal.date) and is_numeric(disposal.premium):
            # 将日期格式和溢价转换为 float
            disposal.date = float(datetime.strptime(disposal.date, "%Y-%m-%d").timestamp())
            disposal.premium = float(disposal.premium)
            
            for location in disposal.location:
                if location is not None and is_numeric(location.area_sqm):
                    # 将 area_sqm 字段转换为 float 类型
                    location.area_sqm = float(location.area_sqm)

                    filtered_disposals.append({
                        'date': disposal.date,
                        'disposal_type': disposal.disposal_type,
                        'lot_no': location.lot_no,
                        'address': location.address,
                        'user': location.user,
                        'area_sqm': location.area_sqm,
                        'premium': disposal.premium
                        })
                    
    disposal_df = pd.DataFrame(filtered_disposals)
    return disposal_df

def select_features(disposal_df):
    # 初始化标签编码器
    label_encoder = LabelEncoder()
    disposal_df['disposal_type'] = label_encoder.fit_transform(disposal_df['disposal_type'])
    disposal_df['lot_no'] = label_encoder.fit_transform(disposal_df['lot_no'])
    disposal_df['address'] = label_encoder.fit_transform(disposal_df['address'])
    disposal_df['user'] = label_encoder.fit_transform(disposal_df['user'])

    # 计算特征之间的相关性
    correlation_matrix = disposal_df.corr()
    
    # 使用热图可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # 选择与目标变量 'premium' 相关性最高的特征
    selected_features = correlation_matrix[abs(correlation_matrix['premium']) > 0.1].index.tolist()
    
    # 移除目标变量 'premium' 自身
    selected_features.remove('premium')

    # 选择与 'premium' 相关性最高的5个特征
    X = disposal_df[selected_features]
    y = disposal_df['premium']
    best_features = SelectKBest(score_func=f_regression, k=5).fit(X, y)
    selected_features_indices = best_features.get_support(indices=True)
    final_selected_features = X.columns[selected_features_indices]
    
    # 输出最终选择的特征
    print("Final Selected Features:", final_selected_features)
    return disposal_df

#建立决策树模型
def model(disposal_df):
    # 划分特征和目标变量
    X = disposal_df[['date', 'disposal_type', 'user', 'area_sqm']]
    y = disposal_df['premium']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化决策树模型
    model = DecisionTreeRegressor()

    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    print("均方误差（MSE）：", mse)
    
    # 输出特征的重要性
    print("特征重要性：", model.feature_importances_)

def _main_():
   json_file_path = 'land-sales.json'
   disposals = read_and_convert_json(json_file_path)
   cleaned_disposals = data_clean(disposals)
#    visualization(cleaned_disposals)
   disposal_df = select_features(cleaned_disposals)
   model(disposal_df)

_main_()