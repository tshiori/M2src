import glob
import numpy as np
from gensim.models import word2vec
import Modules as m
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
#著者名リスト
author_j_name = {"Akutagawa":"芥川",'Arisima':"有島",'Kajii':"梶井",'Kikuchi':"菊池",'Sakaguchi':"坂口",'Dazai':"太宰",'Nakajima':"中島",'Natsume':"夏目",'Makino':"牧野",'Miyazawa':"宮沢"}
#authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
authors = ['Akutagawa','Sakaguchi','Miyazawa']
#authors = ['Akutagawa','Sakaguchi']
#authors = ['Miyazawa']
#グラフ用カラーリスト
color_list = ['red','hotpink','darkorange','gold','skyblue','green','yellowgreen','brown','blue','black']

##パラメータ設定==============================
#辞書名
dic = dics[3]
#何のファイルか
what = "PCA"
#このプログラムの対象
targets = ["ALL","novel","other"] 
target = targets[0] 
#学習パラメータ
vector = 100
window = 5 
epoc =500 
#使用モデルのが学習したファイル
model_targets = ["ALL","Learn","ALLCorpus","#"]
model_target = model_targets[2] #ALLCorpus
#その他情報(rowなど)
sep = "100"
other = "#"
#拡張子
extension = ".csv"

usemodels = ["each","all"]
usemodel = "each"
cate1=913
cate2=914

##==============================================


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


#全著者モデルの読み込み
model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))


#各著者のインスタンスを生成する
for author in authors:
    if usemodel == "each":
        #各著者モデルの読み込み
        model = word2vec.Word2Vec.load("../model/"+m.nameFile(author,dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
        print( "model = "+m.nameFile(author,dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
    print("=========================="+author+"===============================")
    #各著者のインスタンスを生成する
    auth_works = {}
    index = 0
    auth_works,index_fin = m.getWorkObj(author,index)

    #著者毎の作品名リストを格納するリスト定義
    file_list = {}
    file_list[cate1] = []
    file_list[cate2] = []
    #各作品の重心(特徴)をまとめたリストの初期化
    feature_books = np.empty((0, vector), float)
    novel_feature_books = np.empty((0, vector), float)
    other_feature_books = np.empty((0, vector), float)

    for i in range(0,len(auth_works)):
        #重複なし小説リスト
        if(auth_works[i].duplicate==False and auth_works[i].novel==True):
            file_list[cate1] = m.getFileNameByIdUseGlob(author,dic,auth_works[i],file_list[cate1])
        #重複なし小説以外の作品リスト
        elif(auth_works[i].duplicate==False and auth_works[i].novel==False ):
            file_list[cate2] = m.getFileNameByIdUseGlob(author,dic,auth_works[i],file_list[cate2])


    #小説ファイルの読み込み
    for novel_name in file_list[cate1]:
        #作品の特徴量f(f_ave)の計算
        f_ave = get_f_ave(novel_name)    
        #作品の特徴量を特徴のリストに保存
        novel_feature_books = np.append(novel_feature_books, np.array([f_ave]), axis=0)

    #その他のファイルの読み込み
    for other_name in file_list[cate2]:
        #作品の特徴量f(f_ave)の計算
        f_ave = get_f_ave(other_name)
        #作品の特徴量を特徴のリストに保存
        other_feature_books = np.append(other_feature_books, np.array([f_ave]), axis=0)

    feature_books = np.append(novel_feature_books,other_feature_books, axis=0)
    #print(len(feature_books))

    '''
    print("PCA")
    #主成分分析
    pca = PCA(n_components=2)
    pca.fit(feature_books)
    data_pca = pca.transform(feature_books)
    '''
    #print(len(feature_books))
    #print(len(feature_books[0]))

    #SVM用に整形
    print(str(cate1)+" "+str(len(file_list[cate1])))
    print(str(cate2)+" "+str(len(file_list[cate2])))
    y_novel = [0]*len(file_list[cate1])
    y_other = [1]*len(file_list[cate2])
    y = np.append(y_novel,y_other, axis=0)
    #print(len(y))

    #訓練データと検証データに分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(feature_books, y, train_size=int(len(feature_books)*0.66), random_state=0,shuffle=True)

    #標準化
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_train_std = X_train
    X_test_std = X_test

    '''
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    '''
    from sklearn.svm import SVC
    kernels=["linear","rbf"]
    use_kernel=kernels[0]
    Cnum=1e10
    print("parameter C = "+str(Cnum))
    svm = SVC(kernel=use_kernel, C=Cnum, random_state=0)
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


exit()
# 主成分の寄与率を出力
print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
print("exit")

targets = [cate1,cate2]

#print(len(data_pca))


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
use_kernel=kernels[0]
Cnum=0.1
svm = SVC(kernel=use_kernel, C=Cnum, random_state=0)
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

f = open("../result/accuracy.txt","a")
print(targets,"kernel="+use_kernel,"C="+str(Cnum),file=f)
print('traning data accuracy： %.2f' % accuracy_train,file=f)
print('test data accuracy： %.2f' % accuracy_test,file=f)
print("",file=f)
f.close()

'''
#rbfカーネル
svm_rbf = SVC(kernel="rbf", random_state=0, gamma=0.10, C=10.0)
svm_rbf.fit(X_train_std, y_train)
pred_train = svm_rbf.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print('traning data accuracy： %.2f' % accuracy_train)
'''

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
    for idx, cl in enumerate(np.unique(y)):
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
            elif(idx==1):
                color="aqua"
            plt.scatter(x=X_test[y_test == cl, 0], 
                        y=X_test[y_test == cl, 1],
                        alpha=0.6, 
                        c=color,
                        edgecolor='black',
                        marker=markers[idx], 
                        label="test_"+targets[cl])


plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm,test_idx=range(119,246))

plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.rc('legend', fontsize=16)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('../result/svm_'+str(targets)+'_'+use_kernel+'_C'+str(Cnum)+'.png', dpi=300)
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
