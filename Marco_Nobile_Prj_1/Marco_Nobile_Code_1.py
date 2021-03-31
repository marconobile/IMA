#!/usr/bin/env python
# coding: utf-8

# In[1]:


import javalang
import pandas as pd
import os as os
from os import walk
import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
# import warnings
# warnings.filterwarnings("ignore")


# # Import source directory, parse files and save them in a dataframe

# In[2]:


names = []
num_methods = []
paths_ = []
path = '/Users/marconobile/PycharmProjects/Test/venv/Info modelling project/xerces2-j-trunk/src'

for dirpath, dirnames, filenames in walk(path):
    for file in filenames:
        if file.find(".java") != -1:
            with open(dirpath + "/" + file, 'r') as filehandle:
                filecontent = filehandle.read()
                tree = javalang.parse.parse(filecontent)
                for path, node in tree.filter(javalang.tree.ClassDeclaration):
                    names.append(node.name)
                    num_methods.append(len(node.methods))
                    paths_.append(dirpath + "/" + file)

d = {'class_name': names, 'method_num': num_methods, 'path': paths_}
df = pd.DataFrame(data=d)


# In[3]:


god_class = []
# Let's find the god classes:
for index, row in df.iterrows():
    index_ = df['method_num'].values[index]
    if (df["method_num"].mean() + 6 * np.sqrt(df["method_num"].var()) < index_):
        god_class.append(df.ix[index, :])

god_class_names = [row[0] for row in god_class]

print(god_class)  # here I have class name, n of methods, path
# we have 4 god classes, let's parse them
print((df["method_num"].mean() , np.sqrt(df["method_num"].var())))


# In[4]:


# parsing the files of the god classes
trees = []
for i in range(0, 4):
    f0 = open(god_class[i].path, 'r')
    filecontent0 = f0.read()
    trees.append(javalang.parse.parse(filecontent0).types[0]) 
    f0.close()


# # <span style="color:red">Run all the cells below up to the clustering section for all values (comment/decomment)</span>

# In[5]:


# god_class = trees[3] # do it for 0,1,2,3 (the four god classes)
god_class = trees[0]
# god_class = trees[1]
# god_class = trees[2]
# god_class = trees[3]


# In[6]:


print('This is feature extraction for: ', god_class.name)


# # Let's begin the feature extraction

# In[7]:


def getFields(tree):
    fields_list = []
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if node.name in god_class_names:
            for field in node.fields:
                for declarator in field.declarators:
                    fields_list.append(declarator.name)
    return set(fields_list)  
    # return fields_list of the class


# In[8]:


def getMethods(tree):
    methods_list = []
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if node.name in god_class_names:
            for method in node.methods:
                methods_list.append(method)
    return set(methods_list)  
    # return methods_list of the class


# In[9]:


def getFieldsAccessedByMethod(method, fields_list):
    for _, ref in method.filter(javalang.tree.MemberReference):
        if ref.member in fields_list:
            if method.name in row_methods:
                row_methods[method.name] = row_methods[method.name] + [ref.member]
    return None


# In[10]:


def getMethodsAccessedByMethod(method, list_methods):
    for _, ref in method.filter(javalang.tree.MethodInvocation):
        # if ref.member in list_methods:
        if ref.member in row_methods.keys():
            # if method.name in row_methods:
            if method.name in row_methods.keys():
                row_methods[method.name] = row_methods[method.name] + [ref.member]
    return None


# In[11]:


list_methods = getMethods(god_class) 
#  da fare per tree[0], [1], [2] ,[3]
row_methods = {}
for met in list_methods:
    row_methods[met.name] = []


# In[12]:


fields_list = getFields(god_class)
#  da fare per tree[0], [1], [2] ,[3]
for el in list_methods:
    getFieldsAccessedByMethod(el, fields_list)
    getMethodsAccessedByMethod(el, list_methods)


# In[13]:


list_rows = []
for key, value in row_methods.items():
    row = {}
    for method in list_methods:
        if method.name in value:
            row[method.name]=1
        else:
            row[method.name] = 0
    for field in fields_list:
        if field in value:
            row[field]=1
        else:
            row[field] = 0
    list_rows.append(row)

df = pd.DataFrame(list_rows)


# In[14]:


method_name = []
for met in list_methods:
    method_name.append(met.name)

method_name = list(dict.fromkeys(method_name))

dat1 = pd.DataFrame({'method_name':method_name })

temp=dat1.join(df)
temp = temp.loc[:, (temp != 0).any(axis=0)]
# print(temp) # data frame to be saved


# In[15]:


print('This was feature extraction for: ', god_class.name)


# # <span style="color:red">Export to .csv:</span>

# In[16]:


# temp.to_csv("/Users/marconobile/Desktop/IMA_prj1_csv/"+god_class.name +".csv")


# In[17]:


#######################
# CHECK
# Import datasets
#######################

path1 = "/Users/marconobile/Desktop/IMA_prj1_csv/CoreDocumentImpl.csv"
path2 = "/Users/marconobile/Desktop/IMA_prj1_csv/DTDGrammar.csv"
path3 = "/Users/marconobile/Desktop/IMA_prj1_csv/XIncludeHandler.csv"
path4 = "/Users/marconobile/Desktop/IMA_prj1_csv/XSDHandler.csv"

data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data3 = pd.read_csv(path3)
data4 = pd.read_csv(path4)

#######################
# Fix imported datasets
#######################

data1 = data1.drop(['Unnamed: 0'], axis=1)
data1.update(data1)

data2 = data2.drop(['Unnamed: 0'], axis=1)
data2.update(data2)

data3 = data3.drop(['Unnamed: 0'], axis=1)
data3.update(data3)

data4 = data4.drop(['Unnamed: 0'], axis=1)
data4.update(data4)

print("CoreDocumentImpl", data1.shape)
print("DTDGrammar", data2.shape)
print("XIncludeHandler", data3.shape)
print("XSDHandler", data4.shape)


# # <span style="color:red">Clustering: run for every path (comment/decomment)</span>

# In[18]:


print('Uncomment path for: ', god_class.name)


# In[822]:


# path = "/Users/marconobile/Desktop/IMA_prj1_csv/CoreDocumentImpl.csv"
# path = "/Users/marconobile/Desktop/IMA_prj1_csv/DTDGrammar.csv"
path = "/Users/marconobile/Desktop/IMA_prj1_csv/XIncludeHandler.csv"
# path = "/Users/marconobile/Desktop/IMA_prj1_csv/XSDHandler.csv"
# 
data1 = pd.read_csv(path)
data1 = data1.drop(['Unnamed: 0'], axis=1) # drop index but with method_name
data1_2 = data1.drop(['method_name'], axis =1) # drop method name


# # K-means, evaluate k:

# In[823]:


sil_values = []
k_values= []
for i in range(2,81):
    kmeans = KMeans(n_clusters=i,init = 'k-means++', algorithm = 'full' , random_state=42 ).fit(data1_2.values)
    lab=kmeans.labels_
    sil_values.append(silhouette_score(data1_2.values, labels=lab, metric='euclidean'))
    k_values.append(i)

table=dict(zip(k_values,sil_values))
print(table)
# print('Best k for K-Means :' , max(zip(table.values(), table.keys()))) # 2 clusters should maximize the silohuette value for k means


# In[824]:


table_reversed = {}
rev=sorted(table, key=table.get, reverse=True)
for el in rev:
    table_reversed[el]=table[el]

# print(table_reversed)
maximum_k = max(table_reversed, key=table_reversed.get)  # Just use 'min' instead of 'max' for minimum.

print('Best value of k: ' ,maximum_k, ', with a silhouette value of: ', table_reversed[maximum_k])


# # Hierarchical Clustering:

# In[825]:


dists = ['euclidean', 'l1', 'l2', 'manhattan']
linkages = ['complete', 'average']

results_hie = {}

for k in dists:
    for j in linkages:

        sil_values_1 = []
        k_values_1 = []
        for i in range(2,81):
            hierclust=AgglomerativeClustering(n_clusters=i, affinity=k, linkage=j).fit(data1_2.values)
            lab_1=hierclust.labels_
            sil_values_1.append(silhouette_score(data1_2.values, labels=lab_1, metric= 'euclidean'))
            k_values_1.append(i)
        table_1=dict(zip(k_values_1,sil_values_1))
        print(table_1)
        print('Aglomerative clustering with distance {} and linkage {} :'.format(k,j), max(zip(table_1.values(), table_1.keys())) )
        results_hie[(k,j)] = max(zip(table_1.values(), table_1.keys()))


# In[826]:


sil_values_1 = []
k_values_1 = []
for i in range(2,81):
    hierclust=AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward' ).fit(data1_2.values)
    lab_1=hierclust.labels_
    sil_values_1.append(silhouette_score(data1_2.values, labels=lab_1, metric= 'euclidean'))
    k_values_1.append(i)
table_1=dict(zip(k_values_1,sil_values_1))
print(table_1)
print('Aglomerative clustering with distance euclidean and linkage ward :' ,max(zip(table_1.values(), table_1.keys())) )
results_hie[('euclidean','ward')] = max(zip(table_1.values(), table_1.keys()))


# # Pick the k for k-Means and plug it in:

# In[827]:


print(maximum_k, table_reversed[maximum_k])
kmeans2pred = KMeans(n_clusters= maximum_k, init = 'k-means++', algorithm = 'full', random_state=42 ).fit(data1_2.values)
method_labes_kmean = {}
cluster_id_kmean = set()
for idx,row in data1.iterrows():
    method_labes_kmean[row.values[0]] = kmeans2pred.predict(row.values[1:].reshape(1,-1))[0]
    cluster_id_kmean.add(kmeans2pred.predict(row.values[1:].reshape(1,-1))[0])
# print(method_labes_kmean)
# 45 clusters, from 0 to 44


# ## Let's see how many elements each cluster contains for k-Means:

# In[828]:


tot_clust = set()
for key, value in method_labes_kmean.items():
    tot_clust.add(method_labes_kmean[key])
tot_clust=list(tot_clust)
cluster_el_count = {}
cluster_el_count = dict.fromkeys(tot_clust,0)

################
# for key,value  in method_labes_kmean.items():
#     print("Cluster # {} : {}".format(value, key))
################

for key,value in method_labes_kmean.items():
    for idc in tot_clust:
        if method_labes_kmean[key] == idc:
            cluster_el_count[idc]=cluster_el_count[idc]+1

for key,value in cluster_el_count.items():
    print('ClusterID: ',key, '# of elements: ', cluster_el_count[key])


# In[829]:


print(cluster_el_count)


# # Pick the meth/link for Hie Clu and plug it in:

# In[830]:


maximum = max(results_hie, key=results_hie.get)  # Just use 'min' instead of 'max' for minimum.
print(maximum, results_hie[maximum])


# In[831]:


heiClu2pred = AgglomerativeClustering(n_clusters=results_hie[maximum][1],
                                      affinity=maximum[0], linkage=maximum[1]).fit_predict(data1_2.values)
method_labes_hie = {}
cluster_id_hie = set()
for idx,row in data1.iterrows():
    method_labes_hie[row.values[0]] = heiClu2pred[idx]
    cluster_id_hie.add(heiClu2pred[idx])
print(method_labes_hie)


# ## Let's see how many elements each cluster contains for HieClu:

# In[832]:


tot_clust = set()
for key, value in method_labes_hie.items():
    tot_clust.add(method_labes_hie[key])
tot_clust=list(tot_clust)
cluster_el_count = {}
cluster_el_count = dict.fromkeys(tot_clust,0)

################
# for key,value  in method_labes_kmean.items():
#     print("Cluster # {} : {}".format(value, key))
################

for key,value in method_labes_hie.items():
    for idc in tot_clust:
        if method_labes_hie[key] == idc:
            cluster_el_count[idc]=cluster_el_count[idc]+1

for key,value in cluster_el_count.items():
    print('ClusterID: ',key, '# of elements: ', cluster_el_count[key])


# In[833]:


print(cluster_el_count)


# In[834]:


# so now i have 2 dictionaries: method_labes_hie, 
# and method_labes_kmean with {method_name: cluster_label}


# # Get Intrapairs for the k-Mean clustering:

# In[835]:


def get_clusters_intraparis_kmeans(clusterdict): 
    list_cluster_methods = []
    total_intrapairs = set()
    for id_n in cluster_id_kmean:
        temp_c1 = [k for k, v in clusterdict.items() if v == id_n]  
        list_cluster_methods.append(temp_c1)
        intrapairs = set(itertools.permutations(temp_c1,2))
        total_intrapairs = total_intrapairs.union(intrapairs)
    return list_cluster_methods, total_intrapairs


# In[836]:


_, total_intrapairs_kmean = get_clusters_intraparis_kmeans(method_labes_kmean)
print('Intraparis for k-Means :' ,total_intrapairs_kmean)


# # Get Intrapairs for the Hierarchical clustering:

# In[837]:


def get_clusters_intraparis_hie(clusterdict): 
    list_cluster_methods = []
    total_intrapairs = set()
    for id_n in cluster_id_hie:
        temp_c1 = [k for k, v in clusterdict.items() if v == id_n]  
        list_cluster_methods.append(temp_c1)
        intrapairs = set(itertools.permutations(temp_c1,2))
        total_intrapairs = total_intrapairs.union(intrapairs)
    return list_cluster_methods, total_intrapairs


# In[838]:


_, total_intrapairs_hie = get_clusters_intraparis_hie(method_labes_hie)
print('Intraparis for Hierarchical Clustering :' ,total_intrapairs_hie)


# # Get Intrapairs for ground truth:

# In[839]:


def find_substrings(data):
    gt_names = ['create' , 'object', 'cache' , 'uri', 'standalone', 'encoding' , 'identifier' ,  'user', 'error' ,'content', 'parameter', 'subset' , 'global' , 'component']
    gt = {}
    dfList = list(data['method_name'])

    for m in dfList:
        for k in gt_names:
            if k in m.lower():
                gt[m]=k
                break
            else:
                gt[m]= 'none'
    return gt # format = {method: cluster}


# In[840]:


def get_gt_intraparis(dict_m2l):

    gt_names = {'create': [] , 'object' : [], 'cache' : [], 'uri' : [], 'standalone' : [], 'encoding' : [], 'identifier': [], 'user': [], 'error': [], 'content': [],
                'parameter': [], 'subset': [], 'global': [], 'component': [], 'none': []}
    for key,value in gt_names.items():
        temp=[k for k, v in dict_m2l.items() if v == key]
        gt_names[key]=list(set(temp))

    intraparis = {'create': [], 'object': [], 'cache': [], 'uri': [], 'standalone': [], 'encoding': [], 'identifier': [],
                'user': [], 'error': [], 'content': [],
                'parameter': [], 'subset': [], 'global': [], 'component': [], 'none': []}
    for key,value in gt_names.items():
        result = set(itertools.permutations(gt_names[key],2))
        intraparis[key] = result

    total_set = set()
    for key, value in intraparis.items():
        total_set = total_set.union(value)

    return intraparis ,total_set # dict with clusters/keywords as keys and list of all intraparis: 2D tuples that are the intraparis


# In[841]:


# if commented/uncommented the right dataframe in:
# "Clustering: run for every path (comment/decomment)" , this works
subs_data1= find_substrings(data1)
_ , intraparis_gt_data = get_gt_intraparis(subs_data1)


# # Precision, Recall and F1 measures: 

# # For k-Means:

# In[842]:


precision = len(total_intrapairs_kmean.intersection(intraparis_gt_data))/len(total_intrapairs_kmean)
recall = len(total_intrapairs_kmean.intersection(intraparis_gt_data))/len(intraparis_gt_data)
f1 = (2*precision* recall)/ (precision+recall)
print('K-means with k = {}:\nprecision : {}, recall : {} , F1 : {}'.format(maximum_k,precision,recall,f1))


# # For Hierarchical Clustering:

# In[843]:


precision = len(total_intrapairs_hie.intersection(intraparis_gt_data))/len(total_intrapairs_hie)
recall = len(total_intrapairs_hie.intersection(intraparis_gt_data))/len(intraparis_gt_data)
f1 = (2*precision* recall)/ (precision+recall)
print('Hierarchical Clustering with {} distance and {} linkage and {} clusters:\nprecision = {}, recall = {} , F1 = {}'.format(maximum[0],maximum[1],results_hie[maximum][1],precision,recall,f1))


# # Try a visualization of clusters (iff k<7):

# In[844]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[845]:


# data: data1_2.values


# In[846]:


pca = PCA(n_components=2)
d2=pca.fit_transform(data1_2.values)


# In[847]:


# plt.figure()
# plt.scatter(d2[:,0],d2[:,1], c='red',alpha=0.5)
# plt.show()


# In[848]:


# Visualization for k-Means:
print('K-Means for k = ',maximum_k)


# In[849]:


if maximum_k <7:
    x_clustered = KMeans(n_clusters= maximum_k, init = 'k-means++'
                         , algorithm = 'full' , random_state=42).fit_predict(data1_2.values)
#     x_clustered = kmeans2pred.fit_predict(data1_2.values)
    
    label_color_map = {0:'r', 1: 'g', 2: 'b' , 3 :'y' , 4: 'c', 5: 'm' ,
                      6: 'k'}
    label_color = [label_color_map[l] for l in x_clustered]
    
    unique = list(set(label_color))
    for i, u in enumerate(unique):
        tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
        tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
        plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u)+' = Cluster'+ str(i))    
    plt.legend()
    plt.show()


# # For Hierarchical Clustering:

# In[850]:


# Visualization for Hierarchical Clustering:
print('Hierarchical Clustering for k = ',results_hie[maximum][1])


# In[851]:


if results_hie[maximum][1] <7:
    x_clustered = AgglomerativeClustering(n_clusters=results_hie[maximum][1],
                                      affinity=maximum[0], linkage=maximum[1]).fit_predict(data1_2.values)
 
    label_color_map = {0:'r', 1: 'g', 2: 'b' , 3 :'y' , 4: 'c', 5: 'm' ,
                      6: 'k'}
    label_color = [label_color_map[l] for l in x_clustered]
    
    unique = list(set(label_color))
    for i, u in enumerate(unique):
        tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
        tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
        plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u)+' = Cluster'+ str(i) )    
    plt.xlabel("I PC")
    plt.ylabel("II PC")     
    plt.legend()
    plt.show()

