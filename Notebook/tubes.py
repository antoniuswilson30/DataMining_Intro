import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import linkage

#baca dataset mall customer
df = pd.read_csv('Mall_Customers.csv', delimiter = ',')

#baris dan kolom
df.shape

#ambil 5 teratas
df.head()

#jika ada nilai NA didrop
df = df.dropna()

#melihat min, max, standar deviasi, mean
df.describe()

#mengecek apakah ada data duplikasi atau tidak
df.duplicated()

#mengganti kode Male jadi 1 dan Female jadi 0
df['Gender'].replace(['Male', 'Female'], [1, 0], inplace=True)

#atribut 4 tanpa ID
# dt_4kol = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
dt_2kol = df[['Annual Income (k$)', 'Spending Score (1-100)']]

#numpy array X dari dataframe dt_10kol
array_2kol = np.array(dt_2kol.values)
X = array_2kol


#Melihat perbandingan jenis kelamin
sns.countplot(df['Gender'])

#Scatter Plot untuk melihat pola clustering
sns.scatterplot(df['Annual Income (k$)'], df['Spending Score (1-100)'])
sns.set(rc={'figure.figsize':(5, 5)})
plt.title("Regression Plot")

sns.scatterplot(df['Age'], df['Spending Score (1-100)'])
sns.set(rc={'figure.figsize':(5, 5)})
plt.title("Regression Plot")

sns.scatterplot(df['Gender'], df['Spending Score (1-100)'])
sns.set(rc={'figure.figsize':(5, 5)})
plt.title("Regression Plot")

sns.scatterplot(df['Age'], df['Annual Income (k$)'])
sns.set(rc={'figure.figsize':(5, 5)})
plt.title("Regression Plot")

sns.scatterplot(df['Gender'], df['Annual Income (k$)'])
sns.set(rc={'figure.figsize':(5, 5)})
plt.title("Regression Plot")


#melakukan clustering dengan jumlah cluster 5
kmeans_model = KMeans(n_clusters = 5, random_state = 0).fit(X)


# Simpan hasil clustering berupa nomor klaster tiap objek/rekord di
# variabel klaster_objek
klaster_objek = kmeans_model.labels_

# Simpan hasil clustering berupa centroid (titik pusat) tiap kelompok
# di variabel centroids
centroids = kmeans_model.cluster_centers_

#clustering kmeans dengan pemeriksaan kualitas hasil cluster menggunakan elbow method dan koefisien silhouette
#clustering kmeans dilakukan dengan menjalankan algoritmanya menggunakan nilai k 2 hingga 10
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
    intertia.append(kmeans_model.inertia_)
    score = silhouette_score(X, kmeans_model.labels_,  metric='euclidean')
    silhouette_coefficients.append(score)

#visualisasi hasil elbow method    
plt.plot(K, intertia, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

plt.plot(K, silhouette_coefficients, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel("Silhouette Coefficient")
plt.title("AVG Silhouette Coefficient")
plt.show()


#Lakukan clustering (fit) terhadap X dgn jumlah cluster = 5
kmeans_model = KMeans(n_clusters=5, random_state=0).fit(X)

#Baca rekord-rekord bunga Irish yg belum dikeathui cluster-nya
dt_baru = pd.read_csv('Mall_Customers.csv', delimiter = ',')

#ganti Male jadi 1, Female 0
dt_baru['Gender'].replace(['Male', 'Female'], [1, 0], inplace=True)

#baca data yang atas
dt_baru.head()

#menggunakan 4 atribut
dt_4kol = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#numpy array baru
X_new = np.array(dt_4kol.values)

#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
kmeans_model.predict(X_new)



#mengecek korelasi 
dt_4kol.cov()
korelasi = dt_2kol.corr()
sns.heatmap(data = korelasi, cmap="YlGnBu", linewidths=0.5, linecolor="black")

#melihat korelasi Spending Score dengan atribut lainnya
dt_4kol.corr()['Gender']
dt_4kol.corr()['Age']
dt_4kol.corr()['Annual Income (k$)']
dt_4kol.corr()['Spending Score (1-100)']


#interpresstasi clustering
kmeans_1=KMeans(n_clusters=5)
kmeans_1.fit(X)

y=kmeans_1.predict(X)
df1 = df

df1["label"] = y+1

df1.head()

cluster1=df1[df1["label"]==1]
print('Cluster 1')
print('Jumlah Customer =', len(cluster1))
print('Daftar Customer', cluster1["CustomerID"].values)
print()
cluster2=df1[df1["label"]==2]
print('Cluster 2')
print('Jumlah Customer =', len(cluster2))
print('Daftar Customer', cluster2["CustomerID"].values)
print()
cluster3=df1[df1["label"]==3]
print('Cluster 3')
print('Jumlah Customer =', len(cluster3))
print('Daftar Customer', cluster3["CustomerID"].values)
print()
cluster4=df1[df1["label"]==4]
print('Cluster 4')
print('Jumlah Customer =', len(cluster4))
print('Daftar Customer', cluster4["CustomerID"].values)
print()
cluster5=df1[df1["label"]==5]
print('Cluster 5')
print('Jumlah Customer =', len(cluster5))
print('Daftar Customer', cluster5["CustomerID"].values)

plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label", palette='colorblind', legend='full', data = df1  ,s = 60 )

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Scatter Plot')
plt.show()


print(len(cluster1) / 200 * 100)
print(len(cluster2) / 200 * 100)
print(len(cluster3) / 200 * 100)
print(len(cluster4) / 200 * 100)
print(len(cluster5) / 200 * 100)

annual1 = cluster1['Annual Income (k$)'].values
annual2 = cluster2['Annual Income (k$)'].values
annual3 = cluster3['Annual Income (k$)'].values
annual4 = cluster4['Annual Income (k$)'].values
annual5 = cluster5['Annual Income (k$)'].values

spend1 = cluster1['Spending Score (1-100)'].values
spend2 = cluster2['Spending Score (1-100)'].values
spend3 = cluster3['Spending Score (1-100)'].values
spend4 = cluster4['Spending Score (1-100)'].values
spend5 = cluster5['Spending Score (1-100)'].values

Std_a = np.std(annual1)
mean_a = np.mean(annual1)
print(Std_a)
print(mean_a)
print()
Std = np.std(spend1)
mean = np.mean(spend1)
print(Std)
print(mean)

Std_a = np.std(annual2)
mean_a = np.mean(annual2)
print(Std_a)
print(mean_a)
print()
Std = np.std(spend2)
mean = np.mean(spend2)
print(Std)
print(mean)

Std_a = np.std(annual3)
mean_a = np.mean(annual3)
print(Std_a)
print(mean_a)
print()
Std = np.std(spend3)
mean = np.mean(spend3)
print(Std)
print(mean)

Std_a = np.std(annual4)
mean_a = np.mean(annual4)
print(Std_a)
print(mean_a)
print()
Std = np.std(spend4)
mean = np.mean(spend4)
print(Std)
print(mean)

Std_a = np.std(annual5)
mean_a = np.mean(annual5)
print(Std_a)
print(mean_a)
print()
Std = np.std(spend5)
mean = np.mean(spend5)
print(Std)
print(mean)



#simpan model
pickle.dump(kmeans_model, open('kmeans_model', 'wb'))

#baca model
loaded_model = pickle.load(open('kmeans_model', 'rb'))

#prediksi model
loaded_model.predict(X_new)





####

# Clustering dataset Irish.csv denga algoritma Agglomerative,
# evaluasi hasil pengelompokan dengan komputasi koef Silhoutte,
# pencarian pola untuk tiap kelompok
#
# Oleh: Veronica S. Moertini
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Baca dataset iris.csv
dt_iris = pd.read_csv('iris.csv', delimiter = ',')

#Ambil & print rekord bag atas
print(dt_iris.head())

# Pilih fitur-fitur (4 atribut) yang akan dikelompokkan (tidak 
# mengikutsertakan atribut spesies)
dt_iris_3kol = dt_iris[['sepal_length', 'petal_length', 'petal_width']]
dt_iris_3kol.head()
#dt_iris_4kol.values

# Buat numpy array X dari dataframe dt_iris_4kol
X = np.array(dt_iris_3kol.values)





### EKSPERIMEN UNTUK MENCARI LINKAGE TERBAIK


#### Agglomerative ####

#Complete
K = range(3,8)
for k in K:
    agglo_model_comp = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
    agglo_model_comp.fit_predict(X)
    score = silhouette_score(X, agglo_model_comp.labels_,  metric='euclidean')
    print("k =", k)
    print(score)

#Pilih terbaik dan print dendogram
agglo_model_comp = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
agglo_model_comp.fit_predict(X)
plt.figure(figsize=(15, 10))
plt.title("Dendrogram Mall Customer - Complete Linkage")
dend = sch.dendrogram(sch.linkage(X, method='complete'))
plt.show()

agglo_model_comp.n_clusters
labels = agglo_model_comp.labels_

df_labels = pd.DataFrame({'cls': labels})
df = dt_4kol.join(df_labels)

#Hitung pola tiap kelompok
df_pola = df.groupby(['cls']).describe()
df_pola



#Average
K = range(3,8)
for k in K:
    agglo_model_avg = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')
    agglo_model_avg.fit_predict(X)
    score = silhouette_score(X, agglo_model_avg.labels_,  metric='euclidean')
    print("k =", k)
    print(score)

#Pilih terbaik dan print dendogram
agglo_model_avg = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
agglo_model_avg.fit_predict(X)
plt.figure(figsize=(15, 10))
plt.title("Dendrogram Mall Customer - Average Linkage")
dend = sch.dendrogram(sch.linkage(X, method='average'))
plt.show()

agglo_model_avg.n_clusters
labels = agglo_model_avg.labels_

df_labels = pd.DataFrame({'cls': labels})
df = dt_4kol.join(df_labels)

#Hitung pola tiap kelompok
df_pola = df.groupby(['cls']).describe()
df_pola



#Single
K = range(3,8)
for k in K:
    agglo_model_single = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='single')
    agglo_model_single.fit_predict(X)
    score = silhouette_score(X, agglo_model_single.labels_,  metric='euclidean')
    print("k =", k)
    print(score)

#Pilih terbaik dan print dendogram
agglo_model_single = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
agglo_model_single.fit_predict(X)
plt.figure(figsize=(15, 10))
plt.title("Dendrogram Mall Customer - Single Linkage")
dend = sch.dendrogram(sch.linkage(X, method='single'))
plt.show()


agglo_model_single.n_clusters
labels = agglo_model_single.labels_

df_labels = pd.DataFrame({'cls': labels})
df = dt_4kol.join(df_labels)

#Hitung pola tiap kelompok
df_pola = df.groupby(['cls']).describe()
df_pola



### Cari Pola ###

#complete
df_new = df
models = linkage(df_new, method='complete',metric='euclidean')
labels = cut_tree(models, n_clusters=5).reshape(-1, )
labels

df_new['Cluster_Id'] = labels + 1
df_new.head()

plt.figure(figsize = (20,15))
plt.subplot(7, 5)
sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = df_new,legend='full',palette="Set1")
plt.subplot(7, 5)
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = df_new,legend='full',palette="Set1")
plt.subplot(7, 5)
sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= df_new,legend='full',palette="Set1")
plt.show()

df_new[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()


cluster1=df_new[df_new['Cluster_Id']==1]
print('Daftar Customer', cluster1['CustomerID'].values)
cluster2=df_new[df_new['Cluster_Id']==2]
print('Daftar Customer', cluster2['CustomerID'].values)
cluster3=df_new[df_new['Cluster_Id']==3]
print('Daftar Customer', cluster3['CustomerID'].values)
cluster4=df_new[df_new['Cluster_Id']==4]
print('Daftar Customer', cluster4['CustomerID'].values)
cluster5=df_new[df_new['Cluster_Id']==5]
print('Daftar Customer', cluster5['CustomerID'].values)



#average
df_new = df
models = linkage(df_new, method='average',metric='euclidean')
labels = cut_tree(models, n_clusters=5).reshape(-1, )
labels

df_new['Cluster_Id'] = labels + 1
df_new.head()

plt.figure(figsize = (20,15))
plt.subplot(7, 5)
sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = df_new,legend='full',palette="Set1")
plt.subplot(7, 5)
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = df_new,legend='full',palette="Set1")
plt.subplot(7, 5)
sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= df_new,legend='full',palette="Set1")
plt.show()

df_new[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()


cluster1=df_new[df_new['Cluster_Id']==1]
print('Daftar Customer', cluster1['CustomerID'].values)
cluster2=df_new[df_new['Cluster_Id']==2]
print('Daftar Customer', cluster2['CustomerID'].values)
cluster3=df_new[df_new['Cluster_Id']==3]
print('Daftar Customer', cluster3['CustomerID'].values)
cluster4=df_new[df_new['Cluster_Id']==4]
print('Daftar Customer', cluster4['CustomerID'].values)
cluster5=df_new[df_new['Cluster_Id']==5]
print('Daftar Customer', cluster5['CustomerID'].values)


#single
df_new = df
models = linkage(df_new, method='single',metric='euclidean')
labels = cut_tree(models, n_clusters=5).reshape(-1, )
labels

df_new['Cluster_Id'] = labels + 1
df_new.head()

plt.figure(figsize = (20,15))
plt.subplot(7, 5)
sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = df_new,legend='full',palette="Set1")
plt.subplot(7, 5)
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = df_new,legend='full',palette="Set1")
plt.subplot(7, 5)
sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= df_new,legend='full',palette="Set1")
plt.show()

df_new[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()


cluster1=df_new[df_new['Cluster_Id']==1]
print('Daftar Customer', cluster1['CustomerID'].values)
cluster2=df_new[df_new['Cluster_Id']==2]
print('Daftar Customer', cluster2['CustomerID'].values)
cluster3=df_new[df_new['Cluster_Id']==3]
print('Daftar Customer', cluster3['CustomerID'].values)
cluster4=df_new[df_new['Cluster_Id']==4]
print('Daftar Customer', cluster4['CustomerID'].values)
cluster5=df_new[df_new['Cluster_Id']==5]
print('Daftar Customer', cluster5['CustomerID'].values)