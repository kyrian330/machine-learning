# 根据工资和年龄预测可贷额度
import pandas as pd

# 获取数据源表格里的数据
data = pd.read_csv('loans.csv')
# X数据为工资年龄两列, 此时已构成二维列表, 所以直接通过下标取出即可
X_data = data[['salary', 'age']].values
Y_data = data['loans'].values

# 训练代码不变
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_data, Y_data)

# 预测
Y_pred = regressor.predict(X_data)
print(Y_pred)

# 数据可视化展示
import matplotlib.pyplot as plt
# 绘制工资数据与真实额度的散点图
x = data['salary']
y = Y_pred
plt.plot(x, y, color = 'blue')
plt.show()

# 散点图排序版
# x = sorted(data['salary'])
# y = sorted(Y_pred)
# plt.plot(x, y, color = 'blue')
# plt.show()