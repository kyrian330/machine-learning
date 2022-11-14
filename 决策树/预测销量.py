import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入 Excel文件时要先下载 openpyxl模块 (pip install openpyxl)

path = 'C:/Users/Kyrian/Desktop/PythonCode/PySklearn/决策树/决策树.xlsx'
# data = pd.read_excel(path)
data = pd.read_excel(path, index_col='序号')  # 导入数据
# print(data)

# 数据是类别标签, 要将它转换为数据
# 用 1表示好、是、高这 3个属性, 用 0来表示坏、否、低
data[data == '好'] = 1
data[data == '是'] = 1
data[data == '高'] = 1
data[data != 1] = 0
# 注意 .values.astype(int)不可少, 否则后面 fit()数据的时候会报错 (可尝试无 .values.astype(int)输出, 观察区别)
x = data.iloc[:, :3].values.astype(int)  # 前三列 (自变量)  二维( data.iloc[:, 0:3] )
y = data.iloc[:, 3].values.astype(int)  # 销售列 (因变量)
# print(x)
# print(y)

# 划分数据为 训练集与 测试集, 训练集占 70%, 测试集占 30%
# 划分训练数据
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=13)
# print(X_train)  # 上面四个参数已按设置好的比例随机划分


# 生成决策树模型, 并使用训练数据训练该模型
from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')  # 建立决策树模型, 基于信息上
dtc.fit(X_train, Y_train)


# 进行预测
y_pred = dtc.predict(X_test)
print(y_pred)


# 模型评估
from sklearn.metrics import classification_report
# 输出 预测准确性
print(dtc.score(X_test, Y_test))
# 输出更详细的分类性能
# print(classification_report(y_pred, Y_test, target_names=['销量高', '销量低']))
