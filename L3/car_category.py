# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt

#from sklearn.datasets import make_blobs ##为聚类产生数据集
#from sklearn import metric

#数据加载
#data = pd.read_csv('car_data.csv',encoding="unicode_escape")
data = pd.read_csv('car_data.csv',encoding="gbk")
#print(data.head())

train_x = data[['人均GDP','城镇人口比重','交通工具消费价格指数','百户拥有汽车量']]
#K-Means 手肘法：
# 统计不同K取值的误差平方和
sse = []
for k in range(1, 11):
    # kmeans算法
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    # 计算inertia簇内误差平方和
    sse.append(kmeans.inertia_)
x = range(1, 11)
'''plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()'''

kmeans = KMeans(n_clusters=3)
# 规范化到[0,1]空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
# 使用kmeans进行聚类
kmeans.fit(train_x)
centers=kmeans.cluster_centers_
#print(centers)
predict_y = kmeans.predict(train_x)
data["城市分类"]=predict_y
print(data)

#plt.rcParams['font.sans-serif']='simkai'
#plt.rcParams['axes.unicode_minus']=False
#print(data)
'''mark=['or','ob','og']
for i,d in enumerate(train_x):
    print(d)
    plt.plot(d[0],d[1],mark[predict_y[i]])
plt.show()'''

print('第一类城市为：',data[data["城市分类"]==0]["地区"])
print('第二类城市为：',data[data["城市分类"]==1]["地区"])
print('第三类城市为：',data[data["城市分类"]==2]["地区"])
