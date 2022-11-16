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
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
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


cate1=914
cate2=911

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

#著者毎の作品名リストを格納するリスト定義
file_list = {}
file_list[cate1] = []
file_list[cate2] = []
#各作品の重心(特徴)をまとめたリストの初期化
feature_books = np.empty((0, vector), float)
novel_feature_books = np.empty((0, vector), float)
other_feature_books = np.empty((0, vector), float)
#各著者のインスタンスを生成する
for author in authors:
    print("=========================="+author+"===============================")
    #各著者のインスタンスを生成する
    auth_works = {}
    index = 0
    auth_works,index_fin = m.getWorkObj(author,index)

    for i in range(0,len(auth_works)):
        #重複なし小説リスト
        if(auth_works[i].duplicate==False and auth_works[i].novel==True):
            file_list[cate1] = m.getFileNameByIdUseGlob(author,dic,auth_works[i],file_list[cate1])
        #重複なし小説以外の作品リスト
        elif(auth_works[i].duplicate==False and auth_works[i].novel==False and auth_works[i].category==cate2 ):
            file_list[cate2] = m.getFileNameByIdUseGlob(author,dic,auth_works[i],file_list[cate2])


print(len(file_list[cate1]),len(file_list[cate2]))
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
print(len(novel_feature_books),len(other_feature_books),len(feature_books))

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

from sklearn.model_selection import train_test_split
print(len(feature_books))
X_train, X_test, y_train, y_test = train_test_split(feature_books, y, train_size=int(len(feature_books)*0.66), random_state=0,shuffle=True)

from sklearn.svm import SVC

kernels=["linear","rbf"]
use_kernel=kernels[0]
Cnum=0.1
svm = SVC(kernel=use_kernel, C=Cnum, random_state=0)
svm.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

#traning dataのaccuracy
pred_train = svm.predict(X_train)
accuracy_train = accuracy_score(y_train, pred_train)
print('traning data accuracy： %.2f' % accuracy_train)

#test data の accuracy
pred_test = svm.predict(X_test)
accuracy_test = accuracy_score(y_test, pred_test)
print('test data accuracy： %.2f' % accuracy_test)



