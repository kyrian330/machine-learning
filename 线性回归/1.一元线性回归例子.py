# 根据学习时长预测成绩
import pandas as pd
import numpy as np

# 获取数据源表格里的数据
data = pd.read_csv('studentscores.csv')
X_data = data.loc[:, ['Hours']].values  # 传入学习时长(只取学习时长, 强制转换为二维)
# 或者 X_data = data.iloc[:, 1:2].values
# 或者 X_data = np.array(data.loc[:, 'Hours']).reshape(-1, 1)
Y_data = data['Scores'].values  # 传入学习成绩
print(X_data)

# 导入sklearn的线性回归方法(以下是训练代码)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # 创建 LinearRegre类的对象 regressor
regressor = regressor.fit(X_data, Y_data)  # 用 LinearRegre类的 fit方法对训练集进行训练

# 让上面这个训练好的模型预测一些数据, 使用predi函数
# predict的参数格式与X相同, 必须是二维列表

# 想要预测的数据
test = [[6.8], [1.3], [3.4], [8.8], [9.5]]
# 预测
Y_pred = regressor.predict(test)
print(Y_pred)