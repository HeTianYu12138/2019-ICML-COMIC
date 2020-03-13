import math
import itertools

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from scipy.sparse import csr_matrix, triu, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial import distance

class COMIC:
    def __init__(self, view_size, data_size, k=10, measure='cosine',
                 clustering_threshold=1., eps=1e-5, pair_rate = 0.9, gamma=1,
                 max_iter = 200, verbose=True):

        self.n_samples = data_size
        self.view_size = view_size
        self.k = k  #k in m-knn
        self.measure = measure #m-knn距离计算方式
        self.clustering_threshold = clustering_threshold #voting参数
        self.eps = eps #epsilon参数
        self.verbose = verbose#打印结果
        self.pair_rate = pair_rate#取连接节点的前百分之几，计算epsilon
        self.gamma = gamma#计算L2 Loss参数
        self.max_iter = max_iter#最大迭代次数


        self.labels_ = None #聚类结果
        self.Z = None#映射空间的数据
        self.i = None #mknn邻接矩阵边的一个节点的索引
        self.j = None #另一个节点的索引

    def fit(self, X_list):
        """
        根据数据生成mknn图的边，然后运行COMIC算法
        ----------
        X_list:原数据（shape = Views*n*dim)
        return : 聚类结果[(data_index,label)...]
        """

        assert type(X_list) == list
        assert len(X_list) == self.view_size

        print ('\n*** Compute m-knn graph edges***\n')
        # 计算各视图 m_knn的边 返回结果（i,j)表示xi与xj互相在对方的10最邻近里面
        mknn_list = []
        for view in range(self.view_size):
            print ('compute m-knn graph of view', view+1)
            # 把shape中为1的维度去掉,(1,nums,dims)->(nums,dims)
            X = np.squeeze(X_list[view])
            m_knn_matrix = self.m_knn(X, self.k, measure=self.measure)
            print ('m_knn_matrix', m_knn_matrix.shape)
            mknn_list.append(m_knn_matrix)

        # perform the COMIC clustering
        self.labels = self.run_COMIC(X_list, mknn_list)

        # return the computed labels
        return self.labels

    def pretrain(self, X_list, w_list):
        '''
        初始化超参数 lambda 谱范数之比/mu 相似矩阵W最大边长平方/epsilon/S/Z
        :param X_list: [X1,X2,...Xv] shape:V*N*dim
        :param w_list: [w1,w2,...,wv] 存储连接的节点对
        :return: 初始化的S,Z,W,lambda,mu,epsilon,epsilon_mean?,X的谱范数
        '''
        # preprocess
        # mknn  nodes of edges
        self.i_list = []
        self.j_list = []

        # X谱范数
        xi_list = []
        #mknn W
        weights_list = []
        Z_list = []
        S_list = []
        epsilon_list = []
        lamb_list = []
        # epsilon_mean_list = []
        mu_list = []
        max_iter = self.max_iter

        print ("\n*** Initiation ***\n")
        for view in range(self.view_size):
            X = X_list[view]
            w = w_list[view]
            X = X.astype(np.float32)  # features stacked as N x D (D is the dimension)
            w = w.astype(np.int32)  # list of edges represented by start and end nodes
            # make sure w as size () * 2
            assert w.shape[1] == 2
            
            # initialization
            n_samples, n_features = X.shape
            n_pairs = w.shape[0]
            
            # list of two nodes of edges
            i = w[:, 0]
            j = w[:, 1]

            ########## 初始化W邻接矩阵
            # R [(i,j) 1 (j,i) 1...]
            R = scipy.sparse.coo_matrix((np.ones((i.shape[0] * 2,)),
                                            (np.concatenate([i, j], axis=0),
                                            np.concatenate([j, i], axis=0))), shape=[n_samples, n_samples])
            # mkk 连接边个数 (i,j)与(j,i)各算一个
            n_conn = np.sum(R, axis=1)
            n_conn = np.asarray(n_conn)
            # [m-knn 边的权重]
            weights = np.mean(n_conn) / np.sqrt(n_conn[i] * n_conn[j])
            weights = np.squeeze(weights)
            
            ######### 初始化S,Z
            S = np.ones((i.shape[0],))
            Z = X.copy()

            #########初始化epsilon 90% shortest edges in W
            # 与论文不符之处
            # epsilon = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2 + self.eps, axis=1))
            # Note: suppress low values. This hard coded threshold could lead to issues with very poorly normalized data.
            epsilon = weights;
            epsilon[epsilon / np.sqrt(n_features) < 1e-2] = np.max(epsilon)
            # take the top 90% of the closest neighbours as a heuristic
            # top_samples = np.minimum(250.0, math.ceil(n_pairs * self.pair_rate))
            # epsilon_mean = np.mean(epsilon[:int(top_samples)])
            epsilon = np.sort(epsilon)
            ##########初始化mu
            mu = epsilon[-1] ** 2
            epsilon = np.mean(epsilon[:int(math.ceil(n_pairs * self.pair_rate))])


            # computation of matrix A = D-R (here D is the diagonal matrix and R is the symmetric matrix), see equation (8)

            R = scipy.sparse.coo_matrix((np.concatenate([weights * S, weights * S], axis=0),
                                            (np.concatenate([i, j], axis=0), np.concatenate([j, i], axis=0))),
                                        shape=[n_samples, n_samples])

            D = scipy.sparse.coo_matrix((np.squeeze(np.asarray(np.sum(R, axis=1))),
                                            ((range(n_samples), range(n_samples)))),
                                        (n_samples, n_samples))

            ######## 初始化lambda
            # note: compute the largest magnitude eigenvalue instead of the matrix norm as it is faster to compute
            eigval = scipy.sparse.linalg.eigs(D - R, k=1, return_eigenvectors=False).real
            # precomputing X 谱范数
            xi = np.linalg.norm(X, 2)
            # Calculate lambda as per equation 9.
            lamb = xi / eigval[0]

            if self.verbose:
                print('View', view)
                print('lambda = %.6f, epsilon = %.6f, mu = %.6f' %(lamb, epsilon, mu))
            # save to list
            self.i_list.append(i)
            self.j_list.append(j)
            Z_list.append(Z)
            S = self.to_matrix(S, i, j, (self.n_samples, self.n_samples))
            S_list.append(S)
            epsilon_list.append(epsilon)
            mu_list.append(mu)
            xi_list.append(xi)
            # epsilon_mean_list.append(epsilon_mean)
            lamb_list.append(lamb)
            weights = self.to_matrix(weights, i, j, (self.n_samples, self.n_samples))
            weights_list.append(weights)
        return S_list, Z_list, weights_list, lamb_list, epsilon_list, mu_list, xi_list#, epsilon_mean_list

    def run_COMIC(self, X_list, w_list):
        '''
        COMIC算法 主要函数
        :param X_list: V*n*dim
        :param w_list: V*num_of_edges*2
        :return: 聚类结果
        '''
        max_iter = self.max_iter

        # preprocess S, Z, and so on
        S_list, Z_list, weights_list, lamb_list, epsilon_list, mu_list, xi_list, epsilon_mean_list = self.pretrain(X_list=X_list, w_list=w_list)
        Z_final_concat = np.concatenate((Z_list[:]), axis=1)

        # pre-allocate memory for the values of the objective function
        obj = np.zeros((max_iter,))

        hist_obj = []
        hist_nmi = []
        n_samples = X_list[0].shape[0]

        print('\n*** Training ***\n')
        # start of optimization phase
        for iter_num in range(1, max_iter):
            S_list_old = S_list[:]
            # compute loss.
            obj[iter_num] = self.compute_obj(X_list, Z_list, S_list, lamb_list, mu_list, weights_list, iter_num)
            for view in range(self.view_size):
                X = X_list[view]
                w = w_list[view]
                X = X.astype(np.float32)  # features stacked as N x D (D is the dimension)
                w = w.astype(np.int32)  # list of edges represented by start and end nodes
                n_samples, n_features = X.shape
                n_pairs = w.shape[0]

                i = self.i_list[view]
                j = self.j_list[view]
                
                # update S.
                dist = self.to_matrix(np.sum((Z_list[view][i, :]-Z_list[view][j, :])**2, axis=1), i, j, (self.n_samples, self.n_samples))
                S_list[view] = self.update_S(S_list_old, view, lamb_list[view], mu_list[view], weights_list[view], dist)
                
                # update Z.
                R = weights_list[view] * (S_list[view]**2)
                R = scipy.sparse.coo_matrix(R)
                D = scipy.sparse.coo_matrix((np.asarray(np.sum(R, axis=1))[:, 0], ((range(n_samples), range(n_samples)))),
                                            shape=(n_samples, n_samples))
                L = D-R
                
                M = scipy.sparse.eye(n_samples) + lamb_list[view] * L
                
                # Solve for Z. This could be further optimised through appropriate preconditioning.
                Z_list[view] = scipy.sparse.linalg.spsolve(M, X)

                # update lamb
                eigval = scipy.sparse.linalg.eigs(L, k=1, return_eigenvectors=False).real
                # Calculate lambda as per equation 9.
                lamb = xi_list[view] / eigval[0]
                lamb_list[view] = lamb

            if (abs(obj[iter_num - 1] - obj[iter_num]) < 1e-8):
            # if (abs(obj[iter_num - 1] - obj[iter_num]) < 1e-8) and iter_num > 50:
                print ('Early stop')
                break

        # at the end of the run, assign values to the class members.
        self.Z_list = Z_list
        self.S_list = S_list
        labels = self.compute_assignment(epsilon_list)

        return labels

    def update_S(self, S_list, view_, lamb, mu, weights, dist):
        '''
        更新S
        :param S_list: 前一个迭代的S
        :param view_: 当前的View
        :param lamb: lambda
        :param mu: mu
        :param weights:W(m-knn)邻接矩阵
        :param dist:Zi与Zj间距离
        :return:更新后的S
        '''
        S = 0
        for view in range(self.view_size):
            if view_ == view:
                continue
            S += self.gamma * S_list[view]
        div = self.gamma*(self.view_size - 1) + lamb * weights * dist
        S = (S+mu) / (div+mu)
        return S

    def compute_assignment(self, epsilon_list):
        '''
        根据Z投票，生成聚类结果
        :param epsilon_list: epsilon，投影后的两向量相邻的距离阈值
        :return: 聚类结果（类标签）
        '''
        ret = {}
        is_conn_list = []
        for view in range(self.view_size):
            # computing connected components.
            diff = self.EuclideanDistances(self.Z_list[view], self.Z_list[view])
            is_conn = np.sqrt(diff) <= self.clustering_threshold*epsilon_list[view]
            is_conn = is_conn + 0
            is_conn_list.append(is_conn)
        
        conn = 0
        for is_conn in is_conn_list:
            conn = conn + is_conn
        conn = conn > self.view_size/2

        G = scipy.sparse.coo_matrix(conn)#返回((i,j),True)表示i,j节点连接
        num_components, labels = connected_components(G, directed=False)#连接图统计函数
        ret['vote'] = labels

        return ret

    def compute_obj(self, X_list, Z_list, S_list, lamb_list, mu_list, weights_list, iter_num):
        '''
        计算Loss损失值
        :param X_list:原数据
        :param Z_list:投影数据
        :param S_list:相似度矩阵
        :param lamb_list:lambda
        :param mu_list:mu
        :param weights_list:W（m_knn)邻接矩阵
        :param iter_num:迭代次数
        :return:Loss
        '''
        # L_1
        L1 = 0
        l1 = 0
        l2 = 0
        for view in range(self.view_size):
            l1 += 0.5 * np.mean(np.sum((X_list[view] - Z_list[view])**2, axis=1))
            i = self.i_list[view]
            j = self.j_list[view]
            dist = self.to_matrix(np.sum((Z_list[view][i, :]-Z_list[view][j, :])**2, axis=1), i, j, (self.n_samples, self.n_samples))
            dot = weights_list[view] * (S_list[view]**2) * dist
            l2 += 0.5 * lamb_list[view] * (np.mean(dot)+ mu_list[view] * np.mean((S_list[view]-1)**2))
        L1 = (l1+l2) / self.view_size

        # L_2
        L2 = 0
        # generate permutation of different views
        ls = itertools.permutations(range(self.view_size), 2)
        
        for (view_i, view_j) in ls:
            L2 += 0.5 * np.mean((S_list[view_i]-S_list[view_j])**2)
        L2 = L2 * 0.5
        # final objective
        loss = L1 + self.gamma * L2

        if self.verbose:
            print('iter: %d,  loss: %.20f' %(iter_num, loss))
        return loss

    def EuclideanDistances(self, A, B):
        '''
        计算A （m个列向量）,B （n个列向量）两个矩阵的列向量间的距离
        ED(结果）ij = Ai与Bj间的距离
        :param A:
        :param B:
        :return:
        '''
        BT = B.transpose()
        vecProd = np.dot(A,BT)
        SqA =  A**2

        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

        SqB = B**2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))

        SqED = sumSqBEx + sumSqAEx - 2*vecProd
        SqED[SqED<0]=0.0
        ED = np.sqrt(SqED)
        return ED

    def to_matrix(self, S_, i, j, shape, is_symmetric=True):
        '''
        根据指定的行和列索引将向量数据转换成矩阵
        :param S_: 需要转化的矩阵
        :param i:需要转化的行
        :param j:需要转化的列
        :param shape: 结果矩阵的形状
        :param is_symmetric: 是否为对称
        :return:
        '''
        S = np.zeros(shape)
        for cnt in range(len(S_)):
            if is_symmetric:
                S[i[cnt], j[cnt]] = S_[cnt]
                S[j[cnt], i[cnt]] = S_[cnt]
            else:
                S[i[cnt], j[cnt]] = S_[cnt]
        return S

    @staticmethod
    def m_knn(X, k, measure='cosine'):
        '''
        生成相互连接的节点对
        :param X: X=[x1,x2,x3,...,xn]^T shape：n*dim
        :param k: k临近
        :param measure:两向量距离计算方式
        :return:[[i,j],[],[],...,[]]表示i与j 互相在对方的k临近里，即i与j连接
        '''
        samples = X.shape[0]  # n samples
        print(samples)
        batch_size = 10000  # step
        b = np.arange(k + 1)
        # tuple 不能增删改，效率更高；set不能重复，可用于去重
        b = tuple(b[1:].ravel())  # b = (1,2,3,4,5,6,7,8,9,10)
        z = np.zeros((samples, k))  # z (n,10);每个点的10 nearest points;
        weigh = np.zeros_like(z)  # w (n,10);每个点的10 nearest points的权重
        # X = nums*dim
        for x in np.arange(0, samples, batch_size):
            # np.arange(start,end,step)
            start = x
            end = min(x + batch_size, samples)
            # 计算两集合各元素组合的pairwise距离
            # wij = x1数组中第i个元素点与x2数组中第j个元素点的距离
            w = distance.cdist(X[start:end], X, measure)

            # argpartition只排序指定参数（第k个），其他的位置的值不保证排序正确
            y = np.argpartition(w, b, axis=1)

            z[start:end, :] = y[:, 1:k + 1]
            weigh[start:end, :] = np.reshape(w[tuple(np.repeat(np.arange(end - start), k)),
                                               tuple(y[:, 1:k + 1].ravel())], (end - start, k))
            del w

        ind = np.repeat(np.arange(samples), k)

        P = csr_matrix((np.ones((samples * k)), (ind.ravel(), z.ravel())), shape=(samples, samples))
        Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))

        Tcsr = minimum_spanning_tree(Q)
        P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
        P = triu(P, k=1)

        V = np.asarray(find(P)).T
        return V[:, :2].astype(np.int32)