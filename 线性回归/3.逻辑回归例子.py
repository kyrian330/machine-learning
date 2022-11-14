# 逻辑回归（Logistic Regression）：是一个分类模型，通常用于二分类
# 二分类问题：自变量有一个或多个，因变量的结果只有两种，通常为0和1

import pandas as pd

# 读取年龄、月收入作为X数据(二维), 是否买车作为Y数据
data = pd.read_csv('car.csv')
X_data = data[['age', 'salary']].values
Y_data = data['car'].values

# 导入逻辑回归类, 并创建一个逻辑回归对象
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_data, Y_data)  # 给此对象传入X、Y数据进行学习

# 拟一些测试数据, 让模型判断是否买车
testX = [[28, 8], [30, 10], [32, 5]]
Y_pred = lr.predict(testX)
print("0表示未买车, 1表示已买车:", Y_pred)  # [1 1 0]
# 使用predi_proba函数可以得到没买车和买车的概率
y_pred = lr.predict_proba(testX)
print("没买车和买车的概率:\n", y_pred)

