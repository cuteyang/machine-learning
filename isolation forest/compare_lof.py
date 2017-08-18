import numpy as np
import matplotlib.pyplot as p
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import *
from sklearn.preprocessing import Imputer
import pandas as pd

read_data = pd.read_csv('breast-cancer-wisconsin.csv',header=None,na_values='?')  #从文件读出来的数据
read_data.mean()
data = np.zeros((read_data.shape[0],read_data.shape[1]-2))
data = read_data.iloc[:,1:10].values        #提取其中用于操作的特征
id_per = read_data.iloc[:,0]                #病人编号
classify = read_data.iloc[:,10].values      #病人分类 2为良性，4为恶性
data=Imputer().fit_transform(data)          #对nan求当前列的均值
pca = PCA(n_components=3)                   #此处PCA降维后的特征数量取3，主要是通过22行计算累计方差百分比已超过80%，且3维数据比较好作图展示
pca_data = pca.fit_transform(data)          #对data降维
print(sum(pca.explained_variance_ratio_))
rng = np.random.RandomState(42)
def iforest(pca_data):
    clf = IsolationForest(max_samples=100, random_state=rng, contamination=0.015)
    clf.fit(pca_data)  # 孤立森林算法
    y_pred = clf.predict(pca_data)
    tmp = clf.decision_function(pca_data)
    # abnormal_scores = MinMaxScaler().fit_transform(tmp) #异常得分情况并做归一化处理，越小越异常
    abnormal_scores = tmp
    result = pd.concat([read_data, pd.Series(abnormal_scores)], axis=1, ignore_index=True)
    result.rename_axis({11: "abnormal_scores"}, axis="columns", inplace=True)
    # print(result)
    output_result = result.sort_values(by="abnormal_scores").head(10)
    print(output_result)
    with open("outlier_result.csv",'w') as f:
        f.writelines('****iforest****\n')
    output_result.to_csv("outlier_result.csv", encoding='utf-8',mode='a')  # 存储最有可能为异常点的10个数据
    return y_pred

def lof(pca_data):
    clf = LocalOutlierFactor(n_neighbors=350,contamination=0.015)
    y_pred = clf.fit_predict(pca_data)
    abnormal_scores = clf.negative_outlier_factor_      #outliers tend to have a larger LOF score
    result = pd.concat([read_data, pd.Series(abnormal_scores)], axis=1, ignore_index=True)
    result.rename_axis({11: "abnormal_scores"}, axis="columns", inplace=True)
    output_result = result.sort_values(by="abnormal_scores",ascending=True).head(10)
    with open("outlier_result.csv",'a') as f:
        f.writelines('****lof****\n')
    output_result.to_csv("outlier_result.csv", encoding='utf-8',mode='a')  # 存储最有可能为异常点的10个数据
    print(output_result)
    return y_pred

y_pred1 = iforest(pca_data)
y_pred2 = lof(pca_data)
fig = p.figure(figsize=(16,8))#开始画图
ax = fig.add_subplot(121,projection = '3d')
p.title('iforest')
ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
for i,instances in enumerate(pca_data):
    color ="#FF0000" if y_pred1[i] == -1 else "#5CACEE"
    ax.scatter(instances[0],instances[1],instances[2],c=color,s=20,edgecolor = '#0A0A0A')
ax = fig.add_subplot(122,projection = '3d')
p.title('lof')
ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
for i,instances in enumerate(pca_data):
    color ="#FF0000" if y_pred2[i] == -1 else "#5CACEE"
    ax.scatter(instances[0],instances[1],instances[2],c=color,s=20,edgecolor = '#0A0A0A')
p.show()
