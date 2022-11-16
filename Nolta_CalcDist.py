from pickle import FALSE
import sys
import glob
import numpy as np
from gensim.models import word2vec
import Modules as m
import numpy as np

'''
第二引数で推定元の著者名(main_author)を指定する
辞書は変数にいれるやつ普通に書き変えて
'''

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
##重心・分散共分散を出力するかどうか
save_ave = False #False->出力しない True->出力する
save_cov = False

##パラメータ設定==============================
#辞書名
dic = dics[3] #uNEologd
#何のファイルか
what = "Mahala"
#このプログラムの対象
targets = ["ALL","Learn","Test","ALLCorpus"] 
target = targets[3] #ALLCorpus
#学習パラメータ
vector = 100
window = 5
epoc = 500

#著者リストの全小説作品リスト内の単語に対するone-hotベクトル辞書の作成
#使用モデルのが学習したファイル
model_targets = ["*","ALL","Learn","ALLCorpus"]

#その他情報(rowなど)
sep = 100
other = "alpha-worker1"
#拡張子
extension = ".csv"
##==============================================

##その他初期化
#カウント用任意変数
i=0
#作品の重心（特徴f(A,α)）の初期化
f_ave = np.zeros(vector)
#各作品の重心(特徴)をまとめたリストの初期化
test_feature_books = np.empty((0, vector), float)
#Test用ファイルの著者を保存するリストの初期化
author_books = []
#メイン著者作品の重心(特徴)をまとめたリストの初期化
main_feature_books = np.empty((0, vector), float)
#テスト用作品の全ファイル名を保存するリストの初期化
test_list = []

###ターミナル入力----------------------------------------------------------------------------------------------

#著者名リスト
#authors = ['Akutagawa','Sakaguchi','Dazai','Makino','Miyazawa']
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']


#第二引数は推定元の著者名
main_author = sys.argv[1]
if not (main_author in authors):
    print("[ERROR] author name is Wrong")
    print(authors)
    exit(1)
try:
    cate1=int(sys.argv[2])
    print(cate1)
except:
    exit()
try:
    cate2=int(sys.argv[3])
    print(cate2)
except:
    exit()
model_target = "200work"+str(cate1)+str(cate2)
print("model_target",model_target)

###テスト用作品の特徴抽出--------------------------------------------------------------------------------------
#全著者全作品のモデル読み込み
#model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
#print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

#作品の重心を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")


def get_id_list(file_obj):
    import csv
    csv_obj = csv.reader(file_obj,delimiter=",")
    id_list = []
    for line in csv_obj:
        id = line[2]
        id_list.append(id)
    return id_list

def save_ave(author,file_name,f_ave,output_file):
    print(author,file=output_file,end=",")
    f.write(' '.join(list(map(str,f_ave))))
    print(",",file=f,end="")
    print(file_name,file=f)
    


#各著者のインスタンスを生成する
auth_works = {}
index = 0
auth_works,index_fin = m.getWorkObj(main_author,index)

#著者毎の作品名リストを格納するリスト定義
file_list = {}
file_list[cate1] = []
file_list[cate2] = []

for i in range(0,len(auth_works)):
    #重複なし小説リスト
    if(auth_works[i].duplicate==False and auth_works[i].novel==True):
        file_list[cate1] = m.getFileNameByIdUseGlob(main_author,dic,auth_works[i],file_list[cate1])
    #重複なし小説以外の作品リスト
    elif(auth_works[i].duplicate==False and auth_works[i].novel==False and auth_works[i].category==cate2 ):
        file_list[cate2] = m.getFileNameByIdUseGlob(main_author,dic,auth_works[i],file_list[cate2])

#file_list[cate1]とfile_list[cate2]からそれぞれ100作品ずつランダムに選ぶ
import random
for cate in [cate1,cate2]:
    random.seed(42)
    file_list[cate] = random.sample(file_list[cate],100)

file_list_all=file_list[cate1]+file_list[cate2]
print("file_list_all length = ",len(file_list_all))


#各作品の重心(特徴)をまとめたリストの初期化
feature_books = np.empty((0, vector), float)
novel_feature_books = np.empty((0, vector), float)
other_feature_books = np.empty((0, vector), float)
#各著者モデルの読み込み
model_name = m.nameFile(main_author,dic,"#","200work"+str(cate1)+str(cate2),vector,window,epoc,"#",main_author+"200work"+str(cate1)+str(cate2),other,".model")
model = word2vec.Word2Vec.load("../model/"+model_name)
print( "model = "+model_name)
for file_path in file_list[cate1]:
        #作品の特徴量f(f_ave)の計算
        f_ave = m.get_f_ave(file_path,model,vector)
        #作品の特徴量を特徴のリストに保存
        novel_feature_books = np.append(novel_feature_books, np.array([f_ave]), axis=0)
for file_path in file_list[cate2]:
        #作品の特徴量f(f_ave)の計算
        f_ave = m.get_f_ave(file_path,model,vector)
        #作品の特徴量を特徴のリストに保存
        other_feature_books = np.append(other_feature_books, np.array([f_ave]), axis=0)

feature_books = np.append(novel_feature_books,other_feature_books, axis=0)

'''
print("PCA")
#主成分分析
pca = PCA(n_components=2)
pca.fit(feature_books)
data_pca = pca.transform(feature_books)
'''
print(len(novel_feature_books))
print(len(other_feature_books))
print(len(feature_books))
#print(len(feature_books[0]))

#SVM用に整形
print(str(cate1)+" "+str(len(file_list[cate1])))
print(str(cate2)+" "+str(len(file_list[cate2])))
y_novel = [cate1]*len(file_list[cate1])
y_other = [cate2]*len(file_list[cate2])
y = np.append(y_novel,y_other, axis=0)

'''
#訓練データと検証データに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature_books, y, train_size=int(len(feature_books)*0.7), random_state=0,shuffle=True)
'''
#print(len(X_train),len(X_train[0]),X_train)

main_mean = np.mean(novel_feature_books, axis=0)
main_var =  np.var(novel_feature_books, axis=0)
#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/euclid/nolta/"+m.nameFile(main_author,dic,"GenreEuclid",str(cate1)+str(cate2),vector,window,epoc,sep,"#","#",".csv"))
outf = open("../result/euclid/nolta/"+m.nameFile(main_author,dic,"GenreEuclid",str(cate1)+str(cate2),vector,window,epoc,sep,"#","#",".csv"),"a")
for feature,cate in zip(feature_books,y):
    # ユークリッド距離の計算
    dist = np.sqrt(np.sum(np.square(main_mean-feature)/main_var))
    # ファイル書き込み
    print(main_author,file=outf,end=",")
    print(dist,file=outf,end=",")
    print(cate,file=outf)    
outf.close()

#分散共分散行列(cov),逆行列(cov_I)の計算
cov = np.cov(novel_feature_books, rowvar=False,bias=True)
print("calcurated cov, cov.shape()="+str(cov.shape))
cov_I = np.linalg.inv(cov)
print("calcurated cov_I, covI.shape()="+str(cov_I.shape))

#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/euclid/nolta/"+m.nameFile(main_author,dic,"GenreMahala",str(cate1)+str(cate2),vector,window,epoc,sep,"#","#",".csv"))
outf = open("../result/euclid/nolta/"+m.nameFile(main_author,dic,"GenreMahala",str(cate1)+str(cate2),vector,window,epoc,sep,"#","#",".csv"),"a")
for feature,cate in zip(feature_books,y):
    # マハラノビス距離の計算
    dev = feature - main_mean
    #計算
    mahala_dist = np.dot(np.dot(dev.T, cov_I), dev)
    # ファイル書き込み
    print(main_author,file=outf,end=",")
    print(mahala_dist,file=outf,end=",")
    print(cate,file=outf)    
outf.close()
