#Other を　分類911に限定
import sys
import glob
from unicodedata import category
import numpy as np
from gensim.models import word2vec
import Modules as m
import CorpusModule as cm
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt
import math
import csv
import os

vector = 100
targets = ["Novel","Essay"]
category_forwrd_dict = {"911":"Poetry","912":"Play","913":"Novel","914":"Essay","915":"Diary","916":"Record","917":"Proverbs","918":"Works","919":"ChineseWriting"}
category_reverse_dict = {"Poetry":"911","Play":"912","Novel":"913","Essay":"914","Diary":"915","Record":"916","Proverbs":"917","Works":"918","ChineseWriting":"919"}
authors=["Akutagawa","Miyazawa"]

def getWordListFromFileName(filename):
    file = open(filename)
    data = file.read()
    #ファイル内で使用されている単語を1次元リスト化
    wordlist = data.replace("\n"," ").split()
    if len(wordlist) == 0:
        print("[WARNING] File("+filename+") 's word_list length is Zero")
    return wordlist

def get_f_ave(file_name):
    #初期化
    f_ave = np.zeros(vector)
    vocab_vec = np.zeros((vector), float)
    none = 0
    yes = 0
    #作品内で使用されている単語を足し合わせる
    word_list = getWordListFromFileName(file_name)
    for word in word_list:
        try:     
            vocab_vec = np.array(model.wv[word]) 
            yes += 1
        except:
            none += 1
            print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
            exit()
        f_ave += vocab_vec
    #作品の重心（特徴f(A,α)）
    f_ave /= (len(word_list) - none)
    return f_ave

#モデルの読み込み
model_name = "../model/ALLuNeologd-#-tALL-v100-w5-e300-s100-mALLCorpus-#.model"
print("model = "+model_name)
model = word2vec.Word2Vec.load(model_name)

#オブジェクトの読み込み
A_works = cm.getWorkObj(authors[0])
B_works = cm.getWorkObj(authors[1])
#print(works[0].PrintSelfInformation())
#print(works[1].PrintSelfInformation())
#print(works[370].PrintSelfInformation())


#著者毎の作品名リストを格納するリスト定義
file_list = {}
#各作品の重心(特徴)をまとめたリストの初期化
feature_books = np.empty((0, vector), float)
novel_feature_books = np.empty((0, vector), float)
poetry_feature_books = np.empty((0, vector), float)

#小説ファイルをリストアップ
def makeFileList(file_list:dict,listname:str,category:str,works):    
    for i in range(0,len(works)):
        if(str(works[i].category)==category and works[i].duplicate==False):
            file_list[listname].append(works[i].filepath)
    #print(listname,len(file_list[listname]))
    return file_list

file_list[targets[0]]=[]
file_list[targets[1]]=[]
for target in targets:
    file_list = makeFileList(file_list,target,category_reverse_dict[target],A_works)
    file_list = makeFileList(file_list,target,category_reverse_dict[target],B_works) 

#対象ファイルの読み込み
for novel_name in file_list[targets[0]]:
    #作品の特徴量f(f_ave)の計算
    f_ave = get_f_ave(novel_name)    
    #作品の特徴量を特徴のリストに保存
    novel_feature_books = np.append(novel_feature_books, np.array([f_ave]), axis=0)

#対象ファイルの読み込み
for poetry_name in file_list[targets[1]]:
    #作品の特徴量f(f_ave)の計算
    f_ave = get_f_ave(poetry_name)
    #作品の特徴量を特徴のリストに保存
    poetry_feature_books = np.append(poetry_feature_books, np.array([f_ave]), axis=0)

feature_books = np.append(novel_feature_books,poetry_feature_books, axis=0)
print(feature_books.shape)

print("PCA")
#主成分分析
pca = PCA(n_components=2)
pca.fit(feature_books)
data_pca = pca.transform(feature_books)

# 主成分の寄与率を出力
print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

'''
print("plot PCA")
#プロット
plt.rcParams['font.family'] = 'IPAexGothic'
plt.tick_params(labelsize=18)
plt.figure(figsize=(20, 12), dpi=300)
plt.rcParams["font.size"] = 40
num = 0
x = []
y = []
x_max = -999
y_max = -999
x_min = 999
y_min = 999

for target in targets:
    for file_name in file_list[target]:
        x.append(data_pca[num][0])
        y.append(data_pca[num][1])
        num += 1

    plt.scatter(x, y, s=300, marker="o", label=target)

    if(x_min>min(x)):
        x_min=min(x)
    if(y_min>min(y)):
        y_min=min(y)
    if(x_max<max(x)):
        x_max=max(x)
    if(y_max<max(y)):
        y_max=max(y)
    
    x.clear()
    y.clear()

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.subplots_adjust(left=0.05, right=0.7, bottom=0.05, top=0.95)
#plt.xlim(math.floor(x_min), math.floor(x_max))
#plt.ylim(math.floor(y_min), math.floor(y_max))

title = "全著者全作品モデルから得た芥川作品"+targets[0]+"/"+targets[1]+"の特徴量分布"
#plt.title(title)
plt.xlabel("第一主成分")
plt.ylabel("第二主成分")
#model_target = author+"ALLCorpus"
pca_output_filename="PCA_"+author+"_"+targets[0]+"_"+targets[1]
print("save "+pca_output_filename+".png")
plt.savefig("../result/"+pca_output_filename, bbox_inches='tight')
plt.clf()
plt.cla()

exit()
'''
#print(len(data_pca))

#SVM用に整形
y_novel = [0]*len(file_list[targets[0]])
y_poetry = [1]*len(file_list[targets[1]])
y = np.append(y_novel,y_poetry, axis=0)
#print(len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_pca, y, train_size=200, random_state=0,shuffle=True)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
'''

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.svm import SVC

kernels=["linear","rbf"]
'''
use_kernel=kernels[0]
Cnum=10
svm = SVC(kernel=use_kernel, C=Cnum, random_state=0)
svm.fit(X_train_std, y_train)
'''
Cnum=1.0
gammama_num=1.0
#rbfカーネル
use_kernel=kernels[1]

svm = SVC(kernel=use_kernel, random_state=0, gamma=gammama_num, C=Cnum)
svm.fit(X_train_std, y_train)


from sklearn.metrics import accuracy_score

#traning dataのaccuracy
pred_train = svm.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print('traning data accuracy： %.2f' % accuracy_train)

#test data の accuracy
pred_test = svm.predict(X_test_std)
accuracy_test = accuracy_score(y_test, pred_test)
print('test data accuracy： %.2f' % accuracy_test)

f = open("../result/"+author+"_accuracy.txt","a")
print(targets,"kernel="+use_kernel,"C="+str(Cnum),file=f)
if(use_kernel==kernels[1]):
    print("gamma="+str(gammama_num),file=f)
print('traning data accuracy： %.2f' % accuracy_train,file=f)
print('test data accuracy： %.2f' % accuracy_test,file=f)
print("",file=f)
f.close()

#プロット
def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    plt.rcParams['font.family'] = 'IPAexGothic'
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    #
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    count=0
    for idx, cl in enumerate(np.unique(y)):
        if(idx==0):
            count+=1
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label="train_"+targets[cl])
    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')

        else:
            X_test, y_test = X[test_idx, :], y[test_idx]
            print(len(X_test),len(y_test))
        #
        
        for idx, cl in enumerate(np.unique(y_test)):
            print("idx,cl",idx,cl)
            if(idx==0):
                color="fuchsia"
                count+=1
            elif(idx==1):
                color="aqua"
            plt.scatter(x=X_test[y_test == cl, 0], 
                        y=X_test[y_test == cl, 1],
                        alpha=0.6, 
                        c=color,
                        edgecolor='black',
                        marker=markers[idx], 
                        label="test_"+targets[cl])
        print(count)


plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm,test_idx=range(119,236))

plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.rc('legend', fontsize=16)
plt.legend(loc='upper left')
plt.tight_layout()
if(use_kernel==kernels[0]):
    plt.savefig('../result/svm_'+author+'_'+str(targets)+'_'+use_kernel+'_C'+str(Cnum)+'.png', dpi=300)
else:
    plt.savefig('../result/svm_'+author+'_'+str(targets)+'_'+use_kernel+'_C'+str(Cnum)+'_gamma'+str(gammama_num)+'.png', dpi=300)

plt.show()




'''
fig=plt.figure()
plt.rcParams['font.family'] = 'IPAexGothic'
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
for i in range(0,len(X_train_std)):
    print(X_train_std[i])

    try:
        if(y_train[i]==1):
            plt.scatter(X_train_std[i][0],X_train_std[i][1], c='red')
        else:
            plt.scatter(X_train_std[i][0],X_train_std[i][1], c='blue')
    except:
        print(i)
        print(X_train_std[i])

plt.show()
'''