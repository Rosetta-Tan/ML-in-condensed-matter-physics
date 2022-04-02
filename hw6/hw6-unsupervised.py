"""
Unsupervised learning of Ising model (ferromagnetic/paramagnetic phase), include PCA, t-SNE and K-means.
We provide two versions of the code, by writing from scratch (MyXXX) and import from scikit-learn respectively.
Please run the program, tune parameters and plot the figures.
Submit the runtime outputs and the best figures on the course.pku.edu.cn.
Note that you need to submit a total of 8 figures.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


TYPE = "Ising"
np.random.seed(0)
torch.manual_seed(100)


def load_data_wrapper(path):
    amount = 1000
    training_data, _ = np.load(path, allow_pickle=True)
    samples = random.sample(training_data, amount * 2)
    datas, labels = samples[0][0].T, [np.argmax(samples[0][1])]
    for i in range(1, len(samples)):
        datas = np.concatenate((datas, samples[i][0].T), axis=0)
        labels.append(np.argmax(samples[i][1]))
    datas, labels = datas.astype(np.float32), np.array(labels).astype(np.float32)
    np.save("./unsupervised/{}_d.npy".format(TYPE), datas)
    np.save("./unsupervised/{}_l.npy".format(TYPE), labels)
    return datas, labels


class MyPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X):
        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        Y = np.dot(X, M[:, 0:self.n_components])
        return Y.real


class MyTSNE:
    def __init__(self, n_components=2, perplexity=30, early_exaggeration=12, n_iter=1000, lr=200, init='random',
                 weight_decay=0., momentum=0.9):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.init = init
        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.pca = MyPCA(n_components=n_components)

    # 计算欧氏距离
    def cal_distance(self, data):
        assert data.dim() == 2, 'The shape of data should be N*features!'
        r = (data * data).sum(dim=1, keepdim=True)
        D = r - 2 * data @ data.t() + r.t()
        return D

    # 给定距离D与sigma=1/sqrt(beta)，计算pj|i与信息熵H
    def Hbeta(self, D, beta=1.0):
        P = np.exp(np.clip(-D.copy() * beta, a_min=-800, a_max=100))
        sumP = sum(P) + 1e-8
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    # 由距离矩阵计算p(j|i)矩阵，应用二分查找寻找合适sigma
    def p_j_i(self, X, tol=1e-5, perp=30):
        (n, d) = X.shape
        D = self.cal_distance(X).numpy()  # 计算距离矩阵
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perp)
        # 遍历每一个数据点
        for i in range(n):
            # 准备Di
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = 0
            # 开始二分搜索，直到满足误差要求或达到最大尝试次数
            while np.abs(Hdiff) > tol and tries < 50:
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1
            # 最后将算好的值写至P，注意pii处为0
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP
        print("<sigma> = {}".format(np.mean(np.sqrt(1 / beta))))
        return P

    # 计算对称相似度矩阵
    def cal_P(self, data):
        P = self.p_j_i(data, perp=self.perplexity)  # 计算原分布概率矩阵
        P = torch.from_numpy(P).float()  # p_j_i为numpy实现的，这里变回Tensor
        P = (P + P.t()) / P.sum()  # 对称化
        P = P * self.early_exaggeration  # 夸张
        P = torch.max(P, torch.tensor(1e-12))  # 保证计算稳定性
        return P

    # 计算降维后相似度矩阵
    def cal_Q(self, data):
        Q = (1.0 + self.cal_distance(data)) ** -1
        # 对角线强制为零
        Q[torch.eye(data.shape[0], data.shape[0], dtype=torch.long) == 1] = 0
        Q = Q / Q.sum()
        Q = torch.max(Q, torch.tensor(1e-12))  # 保证计算稳定性
        return Q

    # 拟合优化
    def fit_transform(self, X, show=False):
        # 数据预处理
        if type(self.lr) == str:
            self.lr = max(X.shape[0] / self.early_exaggeration / 4, 50)
        if self.init == 'pca':
            Y = torch.from_numpy(self.pca.fit_transform(X)).float()
        else:
            Y = torch.randn(X.shape[0], self.n_components) * 1e-4
        X = torch.from_numpy(X).float()
        Y = nn.Parameter(Y)
        # 先算出原分布的相似矩阵
        P = self.cal_P(X)
        optimizer = optim.SGD([Y], lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        loss_his = []
        for i in range(self.n_iter):
            if i == 100:
                P = P / self.early_exaggeration  # 100轮后取消夸张
            optimizer.zero_grad()
            Q = self.cal_Q(Y)
            loss = (P * torch.log(P / Q)).sum()
            loss_his.append(loss.item())
            loss.backward()
            optimizer.step()
        if show:
            print('final loss={}'.format(loss_his[-1]))
            plt.plot(np.log10(loss_his))
            plt.show()
        return Y.detach().numpy()


class MyKMeans:
    def __init__(self, k):
        self.k = k
        self.cluster_centers_ = None

    # 初始化质心
    def initcluster_centers_(self, data):
        numSample, dim = data.shape
        # k个质心
        self.cluster_centers_ = np.zeros((self.k, dim))
        # 随机选出k个质心
        for i in range(self.k):
            # 随机选取一个样本的索引
            index = int(np.random.uniform(0, numSample))
            # 初始化质心
            self.cluster_centers_[i, :] = data[index, :]

    # 训练
    def fit(self, data):
        # 计算样本个数
        numSample = data.shape[0]
        # 保存样品属性（第一列保存该样品属于哪个簇，第二列保存该样品与它所属簇的误差（该样品到质心的距离））
        clusterData = np.array(np.zeros((numSample, 2)))
        # 确定质心是否需要改变
        clusterChanged = True
        # 初始化质心
        self.initcluster_centers_(data)
        while clusterChanged:
            clusterChanged = False
            # 遍历样本
            for i in range(numSample):
                # 该样品所属簇（该样品距离哪个质心最近）
                minIndex = 0
                # 该样品与所属簇之间的距离
                minDis = 100000.0
                # 遍历质心
                for j in range(self.k):
                    # 计算该质心与该样品的距离
                    distance = np.sqrt(sum((self.cluster_centers_[j, :] - data[i, :]) ** 2))
                    # 更新最小距离和所属簇
                    if distance < minDis:
                        minDis = distance
                        clusterData[i, 1] = minDis
                        minIndex = j
                # 如果该样品所属的簇发生了改变，则更新为最新的簇属性，且判断继续更新簇
                if clusterData[i, 0] != minIndex:
                    clusterData[i, 0] = minIndex
                    clusterChanged = True
            # 更新质心
            for j in range(self.k):
                # 获取样本中属于第j个簇的所有样品的索引
                cluster_index = np.nonzero(clusterData[:, 0] == j)
                # 获取样本中于第j个簇的所有样品
                pointsInCluster = data[cluster_index]
                # 重新计算质心(取所有属于该簇样品的按列平均值)
                self.cluster_centers_[j, :] = np.mean(pointsInCluster, axis=0)
        return clusterData

    # 预测
    def predict(self, datas):
        if self.cluster_centers_ is None:
            self.initcluster_centers_(datas)
        clusterIndexs = []
        for i in range(len(datas)):
            data = datas[i, :]
            # 处理data数据(处理为可以与质心矩阵做运算的形式)
            data_after = np.tile(data, (self.k, 1))
            # 计算该点到质心的误差平方（距离）
            distance = (data_after - self.cluster_centers_) ** 2
            # 计算误差平方和
            erroCluster = np.sum(distance, axis=1)
            # 获取最小值所在索引号,即预测x_test对应所属的类别
            clusterIndexs.append([np.argmin(erroCluster)])
        return np.array(clusterIndexs)


def plotfig_pca(model, outputs, labels):
    plt.figure()
    plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
    plt.title("{}_{}".format(model, TYPE))
    plt.savefig('./unsupervised/{}_{}.png'.format(model, TYPE))
    plt.show()
    plt.close()


def plotfig_tsne(model, outputs, labels, tr_d, tr_l, te_d, te_l):
    plt.figure()
    plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
    plt.title("{}_{}".format(model, TYPE))
    plt.savefig('./unsupervised/{}_{}.png'.format(model, TYPE))
    plt.show()
    plt.close()
    plt.subplot(121)
    for i in range(len(tr_l)):
        plt.scatter(tr_d[i][0], tr_d[i][1], c=('r' if tr_l[i] == 0 else 'b'))
    plt.title("{}_{}_train".format(model, TYPE))
    plt.subplot(122)
    for i in range(len(te_l)):
        plt.scatter(te_d[i][0], te_d[i][1], c=('r' if te_l[i] == 0 else 'b'), marker='x')
    plt.title("{}_{}_test".format(model, TYPE))
    plt.savefig('./unsupervised/{}_{}_train&test.png'.format(model, TYPE))
    plt.show()
    plt.close()


def plotfig_kmeans(model, centers, labels, tr_d, labels_p, te_d, color):
    plt.subplot(121)
    for i in range(len(labels)):
        plt.scatter(tr_d[i][0], tr_d[i][1], c=('r' if labels[i] == color else 'b'))
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='k')
    plt.title("{}_{}_train".format(model, TYPE))
    plt.subplot(122)
    for i in range(len(labels_p)):
        plt.scatter(te_d[i][0], te_d[i][1], c=('r' if labels_p[i] == color else 'b'), marker='x')
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='k')
    plt.title("{}_{}_test".format(model, TYPE))
    plt.savefig('./unsupervised/{}_{}_train&test.png'.format(model, TYPE))
    plt.show()
    plt.close()


def PCADemo(datas, labels):
    clf = PCA(n_components=2)  # sklearn.decomposition.PCA
    outputs = clf.fit_transform(datas)

    clf_ = MyPCA(n_components=2)
    outputs_ = clf_.fit_transform(datas)

    plotfig_pca('PCA', outputs, labels)
    plotfig_pca('MyPCA', outputs_, labels)


def TSNEDemo(datas, labels):
    # parameters
    initial_dims = 50  # 用pca进行数据预处理降到的维度，可选择不进行预处理(None or int)
    perplexity, early_exaggeration = 30, 4
    init = 'pca'  # 降维后的坐标的初始值，可用pca降维结果或随机生成(pca or random)
    n_iter, lr, weight_decay, momentum = 1000, 'auto', 0, 0.5  # lr可选择'auto'

    pca = PCA(n_components=initial_dims)  # sklearn.decomposition.PCA
    clf = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter,
               learning_rate=lr, init=init, method='exact')  # sklearn.manifold.TSNE
    if initial_dims is None:
        outputs = clf.fit_transform(datas)
    else:
        outputs = clf.fit_transform(pca.fit_transform(datas))

    pca_ = MyPCA(n_components=initial_dims)
    clf_ = MyTSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter, lr=lr,
                  init=init, weight_decay=weight_decay, momentum=momentum)
    if initial_dims is None:
        outputs_ = clf_.fit_transform(datas)
    else:
        outputs_ = clf_.fit_transform(pca_.fit_transform(datas))

    length = int(len(labels) / 2)
    tr_d, tr_d_, tr_l = outputs[:length], outputs_[:length], labels[:length]
    te_d, te_d_, te_l = outputs[length:], outputs_[length:], labels[length:]
    plotfig_tsne('TSNE', outputs, labels, tr_d, tr_l, te_d, te_l)
    plotfig_tsne('MyTSNE', outputs_, labels, tr_d_, tr_l, te_d_, te_l)

    return tr_d, tr_d_, te_d, te_d_, tr_l, length


def KMeansDemo(tr_d, tr_d_, te_d, te_d_, tr_l, length):
    k = 2

    clf = KMeans(n_clusters=k)  # sklearn.cluster.KMeans
    clf.fit(tr_d)
    centers = clf.cluster_centers_  # 两组数据点的中心点
    cluster_labels = clf.labels_  # 每个数据点所属分组
    labels_p = clf.predict(te_d)  # 预测
    if sum(abs(tr_l - cluster_labels)) < length / 2:  # 确保类别颜色和tsne图中的一致
        color = 0
    else:
        color = 1

    clf_ = MyKMeans(k)
    clusterData = clf_.fit(tr_d_)
    centers_ = clf_.cluster_centers_  # 两组数据点的中心点
    cluster_labels_ = clusterData[:, 0]  # 每个数据点所属分组
    labels_p_ = clf_.predict(te_d_)  # 预测
    if sum(abs(tr_l - cluster_labels_)) < length / 2:  # 确保类别颜色和tsne图中的一致
        color_ = 0
    else:
        color_ = 1

    plotfig_kmeans('KMeans', centers, cluster_labels, tr_d, labels_p, te_d, color)
    plotfig_kmeans('MyKMeans', centers_, cluster_labels_, tr_d_, labels_p_, te_d_, color_)


if __name__ == "__main__":
    if not os.path.exists('./unsupervised'):
        os.mkdir('./unsupervised')

    if os.path.exists("./unsupervised/{}_d.npy".format(TYPE)):
        datas = np.load("./unsupervised/{}_d.npy".format(TYPE))
        labels = np.load("./unsupervised/{}_l.npy".format(TYPE))
    else:
        datas, labels = load_data_wrapper("{}_dataset.npy".format(TYPE))
    print('datas.shape={}\tlabels.shape={}'.format(datas.shape, labels.shape))

    # PCA dimensionality reduction
    PCADemo(datas, labels)

    # t_SNE dimensionality reduction
    tr_d, tr_d_, te_d, te_d_, tr_l, length = TSNEDemo(datas, labels)

    # K-means clustering
    KMeansDemo(tr_d, tr_d_, te_d, te_d_, tr_l, length)
