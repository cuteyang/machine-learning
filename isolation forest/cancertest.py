import numpy as np
import matplotlib.pyplot as p
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
pca = PCA(n_components=3)
pca_data = pca.fit_transform(data)          #对data降维
print(sum(pca.explained_variance_ratio_))
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng, contamination=0.015)
clf.fit(pca_data)   #孤立森林算法
y_pred = clf.predict(pca_data)
abnormal_scores = MinMaxScaler().fit_transform(clf.decision_function(pca_data)) #异常得分情况并做归一化处理，越小越异常
result = pd.concat([read_data,pd.Series(abnormal_scores)],axis=1,ignore_index=True)
result.rename_axis({11:"abnormal_scores"},axis="columns",inplace=True)
# print(result)
output_result = result.sort_values(by="abnormal_scores").head(10)
print(output_result)
output_result.to_csv("outlier_result.csv")      #存储最有可能为异常点的10个数据

fig = p.figure(figsize=(8,6))#开始画图
ax = fig.add_subplot(111,projection = '3d')
ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
for i,instances in enumerate(pca_data):
    color ="#FF0000" if y_pred[i] == -1 else "#5CACEE"
    ax.scatter(instances[0],instances[1],instances[2],c=color,s=20,edgecolor = '#0A0A0A')
p.show()
