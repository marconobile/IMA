#!/usr/bin/env python
# coding: utf-8

# In[1]:


import javalang
import numpy as np
import os  # import walk
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
# import scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_validate
from random import randint
# import warnings
from sklearn.metrics import confusion_matrix
# warnings.filterwarnings("ignore")
import itertools
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from itertools import product
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


path = '/Users/marconobile/PycharmProjects/Test/venv/Info modelling project/IMA_prj2/jscomp'

x = [(os.path.join(r, file), file[:-len(".java")]) for r, d, f in os.walk(path) for file in f if ".java" in file]


# In[3]:


class_names_final = []
MTH = []
FLD = []
RFC = []
INT = []

SZ = []
CPX = []
EX = []
RET = []

BCM = []
NML = []
WRD = []
DCM = []


# In[4]:


#########################################
# Open, read and parse to compile the AST
#########################################


for javafile, class_name in x: # for each file/class
    # print(javafile, class_name) #323
    with open(javafile, 'r') as filehandle: # open the file in read mode
        filecontent = filehandle.read() # read it
        tree = javalang.parse.parse(filecontent) # parse it
        for _, node in tree.filter(javalang.tree.ClassDeclaration): # then for each node in the tree that is a class declaration node
            if node.name == class_name: # if the name of the class is the name of the file it means that is the "official class"

                SZ_temp = []
                CPX_temp = []
                THR_temp = []
                RET_temp = []
                BCM_temp = []
                num_doc = 0
                tot_num_words = 0

                class_names_final.append(node.name) # let's save the class
                #########################################
                # Start of class metrics:
                #########################################
                MTH.append(len(node.methods)) # save the len of the methods
                FLD.append(len(node.fields)) # save the len of the fields
                if node.implements: # then if a node has an interface
                    INT.append(len(node.implements)) # count them
                else:
                    INT.append(0) # otherwise output 0
                #########################################
                # End of class metrics
                #########################################

                for _, nod in node.filter(javalang.tree.Documented):
                    if nod.documentation:
                        num_doc = num_doc + 1 # freq
                        # words = len(re.findall('\w+', nod.documentation))
                        tot_num_words = tot_num_words + len(re.findall('\w+', nod.documentation))  # val
                        # tot_num_words = tot_num_words+ len(nod.documentation.split()) # val

                WRD.append(tot_num_words)

                BCM.append(num_doc)

                pub_counter = 0  # Init arrays
                call_counter = 0
                avg_len_method_name = 0
                den = 0

                for method in node.methods:
                    stat_counter = 0
                    stm_stats = 0
                    throw_stats = 0
                    ret_stat = 0
                    # avg_len_method_name = 0
                    # den = 0

                    if method.name:
                        avg_len_method_name = avg_len_method_name + len(method.name)
                        den = den + 1

                    if 'public' in method.modifiers:
                        pub_counter = pub_counter + 1

                    for _, met in method.filter(javalang.tree.MethodInvocation):
                        call_counter = call_counter + 1

                    for _, statement in method.filter(javalang.tree.Statement):
                        if type(statement) is not javalang.tree.BlockStatement:
                            stat_counter = stat_counter + 1
                    SZ_temp.append(stat_counter) # number of statements for this method

                    for _, met1 in method.filter(javalang.tree.IfStatement):
                        stm_stats = stm_stats + 1

                    for _, met1 in method.filter(javalang.tree.SwitchStatement):
                        stm_stats = stm_stats + 1

                    for _, met1 in method.filter(javalang.tree.WhileStatement):
                        stm_stats = stm_stats + 1

                    for _, met1 in method.filter(javalang.tree.DoStatement):
                        stm_stats = stm_stats + 1

                    for _, met1 in method.filter(javalang.tree.ForStatement):
                        stm_stats = stm_stats + 1

                    for _, met1 in method.filter(javalang.tree.MethodDeclaration): ## CHECK
                        if met1.throws:
                            throw_stats = throw_stats + len(met1.throws)

                    for _, met1 in method.filter(javalang.tree.ReturnStatement):
                        ret_stat = ret_stat + 1

                    RET_temp.append(ret_stat)
                    THR_temp.append(throw_stats)
                    CPX_temp.append(stm_stats)

                if not CPX_temp:
                    CPX_temp = [0]
                if not SZ_temp:
                    SZ_temp = [0]
                if not THR_temp:
                    THR_temp = [0]
                if not RET_temp:
                    RET_temp = [0]

                if sum(SZ_temp) != 0:
                    DCM.append(tot_num_words/np.sum(SZ_temp))
                else:
                    DCM.append(0)
                if den != 0:
                    NML.append(avg_len_method_name/den)
                else:
                    NML.append(0)

                RET.append(np.amax(RET_temp))
                EX.append(np.amax(THR_temp))
                CPX.append(np.amax(CPX_temp))
                SZ.append(np.amax(SZ_temp))
                RFC.append(call_counter + pub_counter)


# In[5]:


path_target = '/Users/marconobile/PycharmProjects/Test/venv/Info modelling project/IMA_prj2/defects4j-master/framework/projects/Closure/modified_classes'

content = []

for dirpath, dirnames, filenames in os.walk(path_target):
    for file in filenames:
        with open(dirpath + "/" +  file , 'r') as filehandle:
            for l in filehandle:
                # print(l)
            # filecontent =  filehandle.read()
            # print(filecontent)
                content.append(l)
# print(content)

modified_classes=[]
for i in range(len(content)):
#     print(content[i][content[i].rfind(".")+1 : content[i].rfind('\n')])
    modified_classes.append(content[i][content[i].rfind(".") + 1: content[i].rfind('\n')])


target = [0] * len(class_names_final)
map_var_tar = {}

for i in range(len(class_names_final)):
    for j in range(len(modified_classes)):
        if class_names_final[i] == modified_classes[j]:
            map_var_tar[class_names_final[i]] = 1
            # print(class_names_final)
            target[i] = 1
        else:
            map_var_tar[class_names_final[i]] = 0
# print(map_var_tar)


# In[6]:


d = {"CLASS NAME: ":class_names_final, 'MTH': MTH , "FLD": FLD, "RFC": RFC ,"INT": INT , "SZ" : SZ , "CPX": CPX, "EX": EX, "RET" : RET ,"BCM":BCM , "NML": NML, "WRD": WRD, "DCM":DCM, "buggy": target}
df = pd.DataFrame(data=d)

# print(df.shape)
# print(df.describe()) # input dataframe
print(df.describe(include=['object']))
# df.to_csv("/Users/marconobile/Desktop/feature_vectors.csv")


# In[7]:


df.describe()


# In[8]:


print(df.head())


# In[9]:


df.describe()


# In[10]:


group_by_bug = df.groupby(['buggy'])
group_by_bug['WRD','DCM'].describe()


# In[11]:


# df['buggy']
# Plot of target variable for the training dataset:
df['buggy'].hist(weights = np.ones_like(df['buggy'])*100 / len(df['buggy']), bins=3,alpha=1)
plt.xlabel('TargetVariable')
plt.ylabel('Percentage')
print("TargetVariable frequency values :\n", df['buggy'].value_counts()) # Frequencies of Binary output 
print("TargetVariable percentage values:\n", df['buggy'].value_counts()*100/len(df['buggy'])) # Frequencies of Binary output 


# In[12]:


import seaborn as sns
p=sns.pairplot(df)


# # Let's train the classifiers:

# In[13]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## 1. Decision Tree 

# In[14]:


# Let's split data into X and y:


# In[15]:


X = df[['MTH', 'FLD', 'RFC', 'INT', 'SZ', 'CPX', 'EX', 'RET',
       'BCM', 'NML', 'WRD', 'DCM']]
# print(X.head())


# In[16]:


y = df[['buggy']]
# print(y.head())


# In[17]:


# Now let's split dataset in training and test set 
# Use 80% of dataset as train set and 20% as test set


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[19]:


# X_test.shape


# In[20]:


# Feature Scaling:
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[21]:


# X_test already scaled
pca = PCA(n_components=2)
d2 = pca.fit_transform(X_test)


# ## Decision Tree fitting, tuning and evaluation

# In[22]:


# Let's implement grid search for parameters:


# In[23]:


param_dist = {'max_depth': list(range(1, 10)),
             'min_samples_leaf': list(range(5, 50)),
             'criterion':['gini','entropy'],
             'max_features':['sqrt','log2']}


# In[24]:


# Now we instanciate a decion tree classifier:


# In[25]:


tree = DecisionTreeClassifier(random_state = 42)


# In[26]:


# now we instantiate the randomized search cv which with cv = 10


# In[27]:


tree_cv = GridSearchCV(tree, param_dist ,cv=10, scoring = 'f1', n_jobs=-1)


# In[28]:


# and now we fit the training data:


# In[29]:


tree_cv.fit(X_train,y_train)


# In[30]:


print('### TUNED: ####')
print('f1 score obtained:', tree_cv.best_score_)
print('best parameters:',tree_cv.best_params_)


# In[31]:


# In the following we can see the mean f1 for each cv (with std) 
# for all the combinations of parameter


# In[32]:


# tree_cv.grid_scores_


# In[33]:


# Now we have found the tuned parameters, 
# now let's fit a DecisionTree with the default parameters 
# and let's compare it to our fine tuned model


# In[34]:


# DecisionTree with default
tree_def = DecisionTreeClassifier(random_state = 42)
tree_def.fit(X_train,y_train)
y_pred_def = tree_def.predict(X_test)
cm_def = confusion_matrix(y_test, y_pred_def)
results_def = precision_recall_fscore_support(y_test, y_pred_def, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_def[0],results_def[1],results_def[2]))
print('Confusion Matrix:\n' ,cm_def)
plt.figure()
plot_confusion_matrix(cm_def,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[35]:


# DecisionTree with the parameters found with GridSearch
tree_tuned = DecisionTreeClassifier(criterion = 'entropy',
                                    max_depth = 5, 
                                    min_samples_leaf = 13,
                                    max_features= 'sqrt',
                                    random_state = 42)
# 'criterion': 'entropy', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 7
tree_tuned.fit(X_train,y_train)
y_pred_tun = tree_tuned.predict(X_test)
cm_tun = confusion_matrix(y_test, y_pred_tun)
results_tun = precision_recall_fscore_support(y_test, y_pred_tun, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_tun[0],results_tun[1],results_tun[2]))
print('Confusion Matrix:\n' ,cm_tun)
plt.figure()
plot_confusion_matrix(cm_tun,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[36]:


y_true = np.asarray(y_test)
label = np.absolute(y_true - y_pred_tun)

label_color_map = {0:'r', 1: 'g'}
label_color = [label_color_map[l] for l in label[0]]

unique = list(set(label_color))
for i, u in enumerate(unique):
    tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
    tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
    plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u) + ' (where g = misclassified) ')    
plt.xlabel("I PC")
plt.ylabel("II PC")
plt.legend()
plt.show()


# In[37]:


# DecisionTree with default AND GET RID OF RANDOM STATE
tree_tuned = DecisionTreeClassifier(criterion = 'entropy',
                                    max_depth = 5, 
                                    min_samples_leaf = 13,
                                    max_features= 'sqrt')

tree_tuned.fit(X_train,y_train)


# In[38]:


f1_values_dt = []
precision_dt = []
recall_dt = []
for i in range(20):
    cv_results = cross_validate(tree_tuned, X, y, cv=5,
                                scoring = ('f1','precision','recall'))
    for el in cv_results['test_f1']:
        f1_values_dt.append(el)
    for el in cv_results['test_precision']:
        precision_dt.append(el)
    for el in cv_results['test_recall']:
        recall_dt.append(el)
    
print(len(f1_values_dt),len(precision_dt),len(recall_dt))
# print(f1_values_dt) # just to check if they were random


# # Now time to train GaussianNB:

# In[39]:


from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB.fit(X_train, y_train)
y_pred_NB = clf_NB.predict(X_test)
cm_NB = confusion_matrix(y_test, y_pred_NB)
results_NB = precision_recall_fscore_support(y_test, y_pred_NB, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_NB[0],results_NB[1],results_NB[2]))
print('Confusion Matrix:\n' ,cm_NB)
plt.figure()
plot_confusion_matrix(cm_NB,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[40]:


y_true = np.asarray(y_test)
label = np.absolute(y_true - y_pred_NB)

label_color_map = {0:'r', 1: 'g'}
label_color = [label_color_map[l] for l in label[0]]

unique = list(set(label_color))
for i, u in enumerate(unique):
    tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
    tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
    plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u) + ' (where g = misclassified) ')    
plt.xlabel("I PC")
plt.ylabel("II PC")
plt.legend()
plt.show()


# In[41]:


clf_NB.class_prior_


# In[42]:


f1_values_NB = []
precision_NB = []
recall_NB = []
for i in range(20):
    cv_results = cross_validate(clf_NB, X, y, cv=5,
                                scoring = ('f1','precision','recall'))
    for el in cv_results['test_f1']:
        f1_values_NB.append(el)
    for el in cv_results['test_precision']:
        precision_NB.append(el)
    for el in cv_results['test_recall']:
        recall_NB.append(el)
    
print(len(f1_values_NB),len(precision_NB),len(recall_NB))
# print(f1_values_NB)       # just to check if they were random


# # Training of Linear SVC

# In[43]:


duals = [True,False]
penaltys =  ['l1', 'l2']
losses = ['hinge','squared_hinge']
C =   list(range(1, 50))#[0.01,0.1, 0.5 ,1,50,100,1000] #
all_params = list(product(duals, penaltys, losses, C ))
filtered_params = [{'dual': [dual], 'penalty' : [penalty], 'loss': [loss], 'C':[C]} for dual, penalty, loss, C in all_params
                   if not (penalty == 'l1' and loss == 'hinge')
                   and not ((penalty == 'l1' and loss == 'squared_hinge' and dual is True))
                  and not ((penalty == 'l2' and loss == 'hinge' and dual is False))]


# In[44]:


lin_svm = LinearSVC(random_state = 42)


# In[45]:


lin_svm_cv = GridSearchCV(lin_svm, filtered_params ,cv=10, scoring = 'f1', n_jobs=-1)


# In[46]:


lin_svm_cv.fit(X_train,y_train)


# In[47]:


print('Tuned' , lin_svm_cv.best_params_)


# In[48]:


print( 'TUNED:' )
print('f1 score obtained:', lin_svm_cv.best_score_)
print('best parameters:', lin_svm_cv.best_params_)


# In[49]:


# Now let's try to fit a default SVC and the compare it with the tuned one


# In[50]:


# svm with default
svm = LinearSVC(random_state = 42)
svm.fit(X_train,y_train)
y_pred_def = svm.predict(X_test)
cm_def = confusion_matrix(y_test, y_pred_def)
results_def = precision_recall_fscore_support(y_test, y_pred_def, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_def[0],results_def[1],results_def[2]))
print('Confusion Matrix:\n' ,cm_def)
plt.figure()
plot_confusion_matrix(cm_def,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[51]:


# svm with tuned
svm_tun = LinearSVC(dual= True, loss= 'squared_hinge', 
                penalty= 'l2',C =47 ,random_state = 42)
svm_tun.fit(X_train,y_train)
y_pred_tun = svm_tun.predict(X_test)
cm_tun = confusion_matrix(y_test, y_pred_tun)
results_tun = precision_recall_fscore_support(y_test, y_pred_tun, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_tun[0],results_tun[1],results_tun[2]))
print('Confusion Matrix:\n' ,cm_tun)
plt.figure()
plot_confusion_matrix(cm_tun,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[52]:


y_true = np.asarray(y_test)
label = np.absolute(y_true - y_pred_tun)

label_color_map = {0:'r', 1: 'g'}
label_color = [label_color_map[l] for l in label[0]]

unique = list(set(label_color))
for i, u in enumerate(unique):
    tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
    tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
    plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u) + ' (where g = misclassified) ')    
plt.xlabel("I PC")
plt.ylabel("II PC")
plt.legend()
plt.show()


# In[53]:


# SVM with default AND GET RID OF RANDOM STATE
svm = LinearSVC(dual= True, loss= 'squared_hinge', 
                penalty= 'l2',C =47)

svm.fit(X_train,y_train)


# In[54]:


f1_values_svm = []
precision_svm = []
recall_svm = []

for i in range(20):
    cv_results = cross_validate(svm, X, y, cv=5,
                                scoring = ('f1','precision','recall'))
    for el in cv_results['test_f1']:
        f1_values_svm.append(el)
    for el in cv_results['test_precision']:
        precision_svm.append(el)
    for el in cv_results['test_recall']:
        recall_svm.append(el)
    
print(len(f1_values_svm),len(precision_svm),len(recall_svm))
# print(f1_values_svm) # just to check if they were random


# # Training of MPLClassifier

# In[55]:


from sklearn.neural_network import MLPClassifier


# In[56]:


param_dist = {'hidden_layer_sizes' : list(range(16, 100)),
             'activation': ['logistic', 'tanh', 'relu'],
             'solver' : ['lbfgs'],
#              'learning_rate' : ['constant', 'invscaling', 'adaptive'],
             'early_stopping' : [True], 
             'validation_fraction' : [0.1]}
#[True,False]


# In[57]:


mpl = MLPClassifier(random_state = 42)


# In[58]:


mpl_cv = GridSearchCV(mpl, param_dist ,cv=10, scoring = 'f1', n_jobs= -1)


# In[59]:


mpl_cv.fit(X_train,y_train)


# In[60]:


print('### TUNED: ####')
print('f1 score obtained:', mpl_cv.best_score_)
print('best parameters:', mpl_cv.best_params_)


# In[121]:


# MPL with default
mlp_def = MLPClassifier(random_state = 42,max_iter=500)
mlp_def.fit(X_train,y_train)
y_pred_def = mlp_def.predict(X_test)
cm_def = confusion_matrix(y_test, y_pred_def)
results_def = precision_recall_fscore_support(y_test, y_pred_def, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_def[0],results_def[1],results_def[2]))
print('Confusion Matrix:\n' ,cm_def)
plt.figure()
plot_confusion_matrix(cm_def,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[62]:


# DecisionTree with tuned
mlp = MLPClassifier(activation= 'tanh', early_stopping= True, hidden_layer_sizes= 74,
                    solver= 'lbfgs', validation_fraction= 0.1
                        ,random_state = 42)
# {'activation': 'tanh', 'early_stopping': True, 'hidden_layer_sizes': 74, 'solver': 'lbfgs', 'validation_fraction': 0.1}
mlp.fit(X_train,y_train)
y_pred_tun = mlp.predict(X_test)
cm_tun = confusion_matrix(y_test, y_pred_tun)
results_tun = precision_recall_fscore_support(y_test, y_pred_tun, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_tun[0],results_tun[1],results_tun[2]))
print('Confusion Matrix:\n' ,cm_tun)
plt.figure()
plot_confusion_matrix(cm_tun,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[63]:


y_true = np.asarray(y_test)
label = np.absolute(y_true - y_pred_def)

label_color_map = {0:'r', 1: 'g'}
label_color = [label_color_map[l] for l in label[0]]

unique = list(set(label_color))
for i, u in enumerate(unique):
    tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
    tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
    plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u) + ' (where g = misclassified) ')    
plt.xlabel("I PC")
plt.ylabel("II PC")
plt.legend()
plt.show()


# In[64]:


# mlp with default AND GET RID OF RANDOM STATE
mlp = MLPClassifier()
mlp.fit(X_train,y_train)
# y_pred_def = mlp.predict(X_test)


# In[65]:


f1_values_mlp = []
precision_mlp = []
recall_mlp = []

for i in range(20):
    cv_results = cross_validate(mlp, X, y, cv=5,
                                scoring = ('f1','precision','recall'))
    for el in cv_results['test_f1']:
        f1_values_mlp.append(el)
    for el in cv_results['test_precision']:
        precision_mlp.append(el)
    for el in cv_results['test_recall']:
        recall_mlp.append(el)
    
print(len(f1_values_mlp),len(precision_mlp),len(recall_mlp))
# print(f1_values_mlp) # just to check if they were random


# # Training of RandomForestClassfier

# In[66]:


from sklearn.ensemble import RandomForestClassifier


# In[67]:


param_dist = {'n_estimators' : [8, 16, 32, 64, 100],
             'max_depth': list(range(1, 10)),
             'min_samples_leaf': list(range(5, 20)),
             'criterion':['gini','entropy'],
             'max_features':['sqrt','log2']}


# In[68]:


rf = RandomForestClassifier(random_state = 42)


# In[69]:


rf_cv = GridSearchCV(rf, param_dist ,cv=10, scoring = 'f1', n_jobs= -1)


# In[70]:


rf_cv.fit(X_train,y_train)


# In[71]:


print('TUNED:' )
print('f1 score obtained:', rf_cv.best_score_)
print('best parameters:', rf_cv.best_params_)


# In[72]:


# Random forest with default
rf_def = RandomForestClassifier(random_state = 42)
rf_def.fit(X_train,y_train)
y_pred_def = rf_def.predict(X_test)
cm_def = confusion_matrix(y_test, y_pred_def)
results_def = precision_recall_fscore_support(y_test, y_pred_def, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_def[0],results_def[1],results_def[2]))
print('Confusion Matrix:\n' ,cm_def)
plt.figure()
plot_confusion_matrix(cm_def,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[73]:


# DecisionTree with tuned
rf = RandomForestClassifier(criterion= 'entropy', max_depth= 9, max_features= 'sqrt'
                            ,min_samples_leaf= 7, n_estimators= 8
                        ,random_state = 42)
# best parameters: {'criterion': 'entropy', 'max_depth': 9, 'max_features': 'sqrt', 'min_samples_leaf': 7, 'n_estimators': 8}
rf.fit(X_train,y_train)
y_pred_tun = rf.predict(X_test)
cm_tun = confusion_matrix(y_test, y_pred_tun)
results_tun = precision_recall_fscore_support(y_test, y_pred_tun, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_tun[0],results_tun[1],results_tun[2]))
print('Confusion Matrix:\n' ,cm_tun)
plt.figure()
plot_confusion_matrix(cm_tun,normalize=True, classes=[0, 1],
                      title='Confusion matrix')


# In[122]:


y_true = np.asarray(y_test)
label = np.absolute(y_true - y_pred_def)

label_color_map = {0:'r', 1: 'g'}
label_color = [label_color_map[l] for l in label[0]]

unique = list(set(label_color))
for i, u in enumerate(unique):
    tmp = [d2[:,0][j] for j in range(len(d2)) if label_color[j] == u]
    tmp2 = [d2[:,1][j] for j in range(len(d2)) if label_color[j] == u]
    plt.scatter(tmp, tmp2, c=label_color_map[i], label=str(u) + ' (where g = misclassified) ')    
plt.xlabel("I PC")
plt.ylabel("II PC")
plt.legend()
plt.show()


# In[75]:


# RF with default AND GET RID OF RANDOM STATE
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
# y_pred_def = rf.predict(X_test)


# In[76]:


f1_values_rf = []
precision_rf = []
recall_rf = []

for i in range(20):
    cv_results = cross_validate(rf, X, y, cv=5,
                                scoring = ('f1','precision','recall'))
    for el in cv_results['test_f1']:
        f1_values_rf.append(el)
    for el in cv_results['test_precision']:
        precision_rf.append(el)
    for el in cv_results['test_recall']:
        recall_rf.append(el)
    
print(len(f1_values_rf),len(precision_rf),len(recall_rf))
# print(f1_values_rf) # just to check if they were random


# In[77]:


from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='constant', random_state=0, constant=1)
clf.fit(X, y)
fake_y=clf.predict(X)
results_def = precision_recall_fscore_support(y, fake_y, average='binary')

cm_fake = confusion_matrix(y, fake_y)
# results_tun = precision_recall_fscore_support(y, fake_y, average='binary')
print('Precision: {}, recall: {}, f1 {}'.format(results_def[0],results_def[1],results_def[2]))
print('Confusion Matrix:\n' ,cm_fake)



# In[78]:


f1_values_dc = []
precision_dc = []
recall_dc = []

for i in range(20):
    cv_results = cross_validate(clf, X, y, cv=5,
                                scoring = ('f1','precision','recall'))
    for el in cv_results['test_f1']:
        f1_values_dc.append(el)
    for el in cv_results['test_precision']:
        precision_dc.append(el)
    for el in cv_results['test_recall']:
        recall_dc.append(el)
    
print(len(f1_values_dc),len(precision_dc),len(recall_dc))
# print(f1_values_rf) # just to check if they were random


# # Wilcoxon test on error sets:

# In[79]:


# f1_values_dt, precision_dt, recall_dt
# f1_values_NB, precision_NB, recall_NB
# f1_values_svm, precision_svm, recall_svm
# f1_values_mlp, precision_mlp, recall_mlp
# f1_values_rf, precision_rf, recall_rf


# In[80]:


from scipy.stats import wilcoxon
import itertools


# In[81]:


errors_dict_f1 = {'dt': [f1_values_dt],'NB': [f1_values_NB], 'svm': [f1_values_svm], 
               'mlp' : [f1_values_mlp], 'rf' : [f1_values_rf], 'dc': [f1_values_dc]}


# In[82]:


errors_dict_pr = {'dt': [precision_dt] ,'NB': [precision_NB], 'svm': [precision_svm], 
               'mlp' : [precision_mlp], 'rf' : [precision_rf],'dc': [precision_dc]}


# In[83]:


errors_dict_re = {'dt': [recall_dt] ,'NB': [recall_NB], 'svm': [recall_svm], 
               'mlp' : [recall_mlp], 'rf' : [recall_rf], 'dc': [recall_dc]}


# In[84]:


comb = list(itertools.combinations(list(errors_dict_f1.keys()),2))
print(comb)


# In[85]:


for tup in comb:
    print('Comparison for the f1 measure:\nTesting {} vs {}'.format(tup[0], tup[1]), ': p-value = ', wilcoxon(errors_dict_f1[tup[0]][0],errors_dict_f1[tup[1]][0],zero_method='wilcox')[1])
    


# In[86]:


for tup in comb:
    print('Comparison for the precison measure:\nTesting {} vs {}'.format(tup[0], tup[1]), ': p-value = ', wilcoxon(errors_dict_pr[tup[0]][0],errors_dict_pr[tup[1]][0],zero_method='wilcox')[1])
    


# In[87]:


for tup in comb:
    print('Comparison for the recall measure:\nTesting {} vs {}'.format(tup[0], tup[1]), ': p-value = ', wilcoxon(errors_dict_re[tup[0]][0],errors_dict_re[tup[1]][0],zero_method='wilcox')[1])
    


# # Descriptive statistics and plots for errors:

# In[88]:


for key, value in errors_dict_f1.items():
    print('Avg f1 for {}: {}, with std: {}'.format(key,round(np.mean(value),3),round(np.std(value),3)))


# In[89]:


for key, value in errors_dict_pr.items():
    print('Avg precision for {}: {}, with std: {}'.format(key,round(np.mean(value),3),round(np.std(value),3)))


# In[90]:


for key, value in errors_dict_re.items():
    print('Avg recall for {}: {}, with std: {}'.format(key,round(np.mean(value),3),round(np.std(value),3)))
    


# In[91]:


p=sns.distplot(errors_dict_f1['dt'][0], kde=True)


# In[92]:


p=sns.distplot(errors_dict_f1['NB'][0], kde=True)


# In[93]:


p=sns.distplot(errors_dict_f1['svm'][0], kde=True)


# In[94]:


p=sns.distplot(errors_dict_f1['mlp'][0], kde=True)


# In[95]:


p=sns.distplot(errors_dict_f1['rf'][0], kde=True)


# In[96]:


# dt, NB, svm, mpl, rf
p=sns.distplot(errors_dict_pr['dt'][0], kde=True)


# In[97]:


p=sns.distplot(errors_dict_pr['NB'][0], kde=True)


# In[98]:


p=sns.distplot(errors_dict_pr['svm'][0], kde=True)


# In[99]:


p=sns.distplot(errors_dict_pr['mlp'][0], kde=True)


# In[100]:


p=sns.distplot(errors_dict_pr['rf'][0], kde=True)


# In[101]:


# dt, NB, svm, mpl, rf
p=sns.distplot(errors_dict_re['dt'][0], kde=True)


# In[102]:


p=sns.distplot(errors_dict_re['NB'][0], kde=True)


# In[103]:


p=sns.distplot(errors_dict_re['svm'][0], kde=True)


# In[104]:


p=sns.distplot(errors_dict_re['mlp'][0], kde=True)


# In[105]:


p=sns.distplot(errors_dict_re['rf'][0], kde=True)


# In[106]:


p = sns.boxplot(errors_dict_f1['dt'][0])


# In[107]:



p = sns.boxplot(errors_dict_f1['NB'][0])


# In[108]:


p = sns.boxplot(errors_dict_f1['svm'][0])


# In[109]:


p = sns.boxplot(errors_dict_f1['mlp'][0])


# In[110]:


p = sns.boxplot(errors_dict_f1['rf'][0])


# In[111]:


# dt, NB, svm, mlp, rf
p = sns.boxplot(errors_dict_pr['dt'][0])


# In[112]:


p = sns.boxplot(errors_dict_pr['NB'][0])


# In[113]:


p = sns.boxplot(errors_dict_pr['svm'][0])


# In[114]:


p = sns.boxplot(errors_dict_pr['mlp'][0])


# In[115]:


p = sns.boxplot(errors_dict_pr['rf'][0])


# In[116]:


p = sns.boxplot(errors_dict_re['dt'][0])


# In[117]:


p = sns.boxplot(errors_dict_re['NB'][0])


# In[118]:


p = sns.boxplot(errors_dict_re['svm'][0])


# In[119]:


p = sns.boxplot(errors_dict_re['mlp'][0])


# In[120]:


p = sns.boxplot(errors_dict_re['rf'][0])


# In[ ]:




