# 导入鸢尾花数据集(利用sklearn库中的load_iris()导入iris数据集)
from sklearn.datasets import load_iris
# 利用train_test_split将iris数据集拆分成训练数据和测试数据
# 这里将20%的数据设置为测试数据
# 为了避免随机拆分造成每次的训练结果不同，将随机种子固定。
from sklearn.model_selection import train_test_split

# 加载数据
iris_dataset = load_iris()

# 取出特征数据
X = iris_dataset['data']
# 取出类别数据
Y = iris_dataset['target']
# 随机拆分数据, 将20%的数据作为测试数据, 并设置固定随机种子
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# 导入k-近邻分类器KNeighborsClassifier, 创建一个对象
from sklearn.neighbors import KNeighborsClassifier
# 创建k-近邻分类器, 设置邻居数为3
knn = KNeighborsClassifier(n_neighbors=3)
# 训练数据, 构建模型(分类器调用fit函数对训练用的x和y数据进行训练)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)

# 将预测类型与真实类型进行比较
# 可使用knn.score函数计算准确率, 参数为测试特征和真实类型

print(y_test)  # 真实类型
print(knn.score(X_test, y_test))  # 计算准确率