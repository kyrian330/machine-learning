# 根据学习时长预测成绩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 获取数据源表格里的数据
data = pd.read_csv('studentscores.csv')
X = data.loc[:, ['Hours']].values  # 传入学习时长
Y = data['Scores'].values  # 传入学习成绩

# 将导入的数据样本分为互不相交的两组———— 训练集和 测试集
# 训练集用来训练模型, 测试集用来评估已训练好的模型对未知数据预测的有效度

# 划分 训练数据和 有效数据
rate = 0.7
num_training = int(rate * len(X))  # 随机抽取 70%的数据作为训练集
num_test = len(X) - num_training  # 剩下 30%数据作为测试集
np.random.seed(0)  # 设置随机种子
# print(num_training)  # 25 * 0.7 ≈ 17
# print(num_test)  # 25 - 17 = 8

# permutation()函数的用法, 打乱原来数据中元素的顺序。
# 1. 输入为整数，返回一个新的打乱顺序的数组
# 2. 输入为数组/list，返回新的顺序被打乱的数组/list
# x = np.random.permutation(10)
# print(x)
# y = np.random.permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(y)

indices = np.random.permutation(len(X))  # indices变成了一个表示 25组'Hours/Scores'数据的随机序号的列表 (主要功能是取原表数据的随机序号)
# print(indices)

# indices[:num_training]表示 indices列表前 18个数据 (注意数据循序已经打乱, 具有随机性)
# X[indices[:num_training]]定位到 csv表的源'Hours'数据
X_train = X[indices[:num_training]]
# print(X_train)
Y_train = Y[indices[:num_training]]  # 取 csv表源 'Scores'数据的 70%
# print(Y_train)
X_test = X[indices[num_training:]]  # 取剩下 30%的随机数据当 测试集
Y_test = Y[indices[num_training:]]


# 导入sklearn的线性回归方法(以下是训练代码)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # 创建 LinearRegre类的对象 regressor
regressor = regressor.fit(X_train, Y_train)  # 用 LinearRegre类的 fit方法对训练集进行训练


# 预测
y_test_pred = regressor.predict(X_test)  # 传入刚划分好的测试集 X_test
print(y_test_pred)


# 对预测的数据进行 可视化
plt.scatter(X_test, Y_test, color='red')  # 原表数据
plt.plot(X_test, y_test_pred, color='black', linewidth='3')  # 模型得出的数据 (和散点重合度越高, 说明模型预测越准确)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
