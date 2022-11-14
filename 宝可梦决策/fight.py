from main import dtc

# HP   攻击   防御   速度    新种族值

# 极端数据
# 1.================================ #
n1 = [100, 100, 100, 100]   # 460
n2 = [99, 120, 120, 99]     # 491
vs = [1, 0, 0, 1]
# ================================== #

# 2.================================ #
n1 = [100, 100, 100, 100]
n2 = [99, 100, 100, 100]
vs = [1, 0, 0, 0]
# ================================== #


def fight():
    p1 = [65, 90, 65, 100]
    p2 = [60, 95, 65, 80]

    player1 = p1[0] * 1.4 + p1[1] * 0.9 + p1[2] * 0.8 + p1[3] * 1.5
    player2 = p2[0] * 1.4 + p2[1] * 0.9 + p2[2] * 0.8 + p2[3] * 1.5

    print("player1: ", player1)
    print("player2: ", player2)


    # 2.自定义数据预测
    X_test_dev = [[1, 0, 0, 1]]
    y_pred_dev = dtc.predict(X_test_dev)
    print("自定义数据预测 y_pred: ", y_pred_dev)


def fightdev(p1, p2):

    player1 = p1[0] * 1.4 + p1[1] * 0.9 + p1[2] * 0.8 + p1[3] * 1.5
    player2 = p2[0] * 1.4 + p2[1] * 0.9 + p2[2] * 0.8 + p2[3] * 1.5

    print("player1: ", player1)
    print("player2: ", player2)

    # 2.自定义数据预测
    X_test_dev = [[1, 0, 0, 1]]
    y_pred_dev = dtc.predict(X_test_dev)
    print("自定义数据预测 y_pred: ", y_pred_dev)


p1 = [65, 90, 65, 100]
p2 = [60, 95, 65, 80]
fightdev(p1, p2)