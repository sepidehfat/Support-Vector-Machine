import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TEST_RATIO = 0.2
FEATURE_NOM = 2
C = 10000
learning_rate = 0.000001
max_epochs = 5000


def generate_data():
    x1 = np.random.uniform(0, 1.5, 150).reshape(150, 1)
    x2 = np.random.uniform(0, 1.5, 150).reshape(150, 1)
    x3 = np.random.uniform(2, 3.5, 150).reshape(150, 1)
    x4 = np.random.uniform(2, 3.5, 150).reshape(150, 1)

    y_1 = np.ones((150, 1))
    y_2 = np.ones((150, 1)) * -1

    x_1 = np.concatenate([x1, x2], axis=1)
    x_2 = np.concatenate([x3, x4], axis=1)

    X = np.concatenate([x_1, x_2], axis=0)
    Y = np.concatenate([y_1, y_2], axis=0)

    X = np.insert(X, 2, np.ones([300]), axis=1)
    data = np.concatenate([X, Y], axis=1)
    data = pd.DataFrame(data)
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def parse_data(data):
    train_size = int(len(data) * (1 - TEST_RATIO))

    X_train = data.loc[0:train_size-1, :FEATURE_NOM].to_numpy()
    Y_train = data.loc[0:train_size-1, FEATURE_NOM+1]

    X_test = data.loc[train_size:, :FEATURE_NOM].to_numpy()
    Y_test = data.loc[train_size:, FEATURE_NOM+1]
    return X_train, np.array(Y_train), X_test, np.array(Y_test)


def cost_gradient(W, x, y):
    sum = np.zeros(len(W))
    distance = 1 - (y * np.dot(x, W))
    if max(0, distance) == 0:
        sum += W
    else:
        sum += W - (C * y) * x
    return sum

def svm(X_train, Y_train):
    weights = np.zeros(X_train.shape[1])
    for epoch in range(1, max_epochs):
        for i in range(len(X_train)):
            temp = cost_gradient(weights, X_train[i], Y_train[i])
            weights = weights - (learning_rate * temp)
    return weights
#%%
def plot_svm(X, Y, W):
    plt.figure()
    plt.title('svm')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='winter')
    x = np.linspace(0, 3.5)
    a = -W[0] / W[1]
    b = +W[2] / W[1]
    y = a * x - b
    plt.plot(x, y, 'r')
    plt.show()
#%%
def accuracy(X_test, Y_test, W):
    Y_predict = []
    for i in range(X_test.shape[0]):
        Y_predict.append(np.sign(np.dot(W, X_test[i])))
    return Y_predict, np.sum(Y_test == Y_predict) / len(Y_test)
#%%
data = generate_data()
X_train, Y_train, X_test, Y_test = parse_data(data)
W = svm(X_train, Y_train)
#%%
plot_svm(X_train, Y_train, W)
#%%
Y_predict, acc = accuracy(X_test, Y_test, W)
print(f'accuracy= {acc*100} %')
