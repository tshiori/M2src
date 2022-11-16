import sys
import glob
import numpy as np
import Modules as m
import numpy as np
from gensim.models import word2vec
import csv

'''
第二引数で推定元の著者名(main_author)を指定する
辞書は変数にいれるやつ普通に書き変えて
'''

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
##重心・分散共分散を出力するかどうか
save_ave = True #False->出力しない True->出力する
save_cov = True

##パラメータ設定==============================
#辞書名
dic = dics[3] #uNEologd
#何のファイルか
what = "Mahala"
#このプログラムの対象
targets = ["ALL","Learn","Test","ALLCorpus"] 
target = targets[0] #ALLCorpus
#学習パラメータ
vector = 100
window = 5 
epoc = 500 
#使用モデルのが学習したファイル
model_targets = ["*","ALL","Learn","ALLCorpus"]
model_target = model_targets[3] # *
#その他情報(rowなど)
sep = "T20L120"
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

cate = 913

### ターミナル入力----------------------------------------------------------------------------------------------

# 著者名リスト
#authors = ['Akutagawa','Sakaguchi','Dazai','Makino','Miyazawa']
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']

# 第二引数は推定元の著者名
main_author = sys.argv[1]
if not (main_author in authors):
    print("[ERROR] author name is Wrong")
    print(authors)
    exit(1)

### テスト用作品の特徴抽出--------------------------------------------------------------------------------------

## 全著者全作品のモデル読み込み
model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

## 作品の重心を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

## 各著者のインスタンスを生成する
all_works = {}
index_fin = 0
last_index = 0
# 全著者のイインスタンスの生成
for author in authors:
    print("=========================="+author+"===============================")
    # 各著者のインスタンスを生成する
    auth_works,index_fin = m.getWorkObj(author,index_fin)
    all_works = {**all_works,**auth_works}
    # 確認とインデックス更新
    try:
        print(all_works[len(all_works)-1].index)
        last_index=index_fin
    except:
        print("!")

# check
print(len(all_works)," object made")
for i in range(0,len(all_works)):
    try:
        all_works[i]
    except:
        print("ERROR")
        exit()



## ターゲットのIDリスト生成
target_f = open("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv"))
L_Ids,T_Ids = m.getIdFromTargetListFile(target_f,authors)
target_f.close

## ターゲットのインデックスリストの取得
# 検証作品のインデックスリストの取得
T_index_list = []
for i in range(0,len(all_works)):
    if(str(all_works[i].id) in T_Ids):
        T_index_list.append(all_works[i].index) 
m.printL(T_index_list)
# 学習作品のインデックスリストの取得
L_index_list = []
for i in range(0,len(all_works)):
    if(str(all_works[i].id) in L_Ids):
        L_index_list.append(all_works[i].index) 
m.printL(L_index_list)


## 検証作品の特徴量を求める
# 特徴保存ファイルの設定
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"EuclidAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"EuclidAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

# 検証作品の重心(特徴)をまとめたリストの初期化
test_feature_books = np.empty((0, vector), float)
# 検証作品の特徴リストの作成とオブジェクト情報の追加
for idx in T_index_list:
    # 作品の特徴量f(f_ave)の計算
    f_ave = m.get_f_ave(all_works[idx].filepath,model,vector)
    all_works[idx].feature = f_ave
    # 作品ファイル名と重心を著者辞書Calc.txtファイル書き込み
    if(save_ave):
        print(all_works[idx].author,file=f,end=",")
        f.write(' '.join(list(map(str,f_ave))))
        print(",",file=f,end="")
        print(all_works[idx].filepath,file=f)
    # 作品の特徴量を特徴のリストに保存
    test_feature_books = np.append(test_feature_books, np.array([f_ave]), axis=0)
if(save_ave):
    f.close


## 学習作品の特徴量を求める
## 特徴保存ファイルの設定
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"MainAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"MainAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

# メイン著者学習作品の重心(特徴)をまとめたリストの初期化
learn_feature_books = np.empty((0, vector), float)
# 学習作品の特徴リストの作成とオブジェクト情報の追加
for index in L_index_list:
    # 作品の特徴量f(f_ave)の計算
    f_ave = m.get_f_ave(all_works[index].filepath,model,vector)
    all_works[index].feature = f_ave
    # 作品ファイル名と重心を著者辞書Calc.txtファイル書き込み
    if(save_ave):
        print(main_author,file=f,end=",")
        f.write(' '.join(list(map(str,f_ave))))
        print(",",file=f,end="")
        print(all_works[index].filepath,file=f)
    # 作品の特徴量を特徴のリストに保存
    learn_feature_books = np.append(learn_feature_books, np.array([f_ave]), axis=0)
if(save_ave):
    f.close


## 平均(main_mean),の計算とファイル書き込み
main_mean = np.mean(learn_feature_books, axis=0)
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"EuclidMainAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"EuclidMainAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")
    print("μ=",file=f,end=",")
    f.write(' '.join(list(map(str,main_mean))))
    print("",file=f)
    f.close


## 学習作品のユークリッド距離の計算
# 作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/euclid/"+m.nameFile(main_author,dic,"WordVectorEuclid",str(cate)+"Learn",vector,window,epoc,sep,"#","#",".csv"))
outf = open("../result/euclid/"+m.nameFile(main_author,dic,"WordVectorEuclid",str(cate)+"Learn",vector,window,epoc,sep,"#","#",".csv"),"a")
for index in L_index_list:
    # ユークリッド距離の計算
    dist = np.sqrt(np.sum(np.square(main_mean-all_works[index].feature)))
    # ファイル書き込み
    print(main_author,file=outf,end=",")
    print(dist,file=outf,end=",")
    print(all_works[index].filepath,file=outf)
outf.close


## 検証作品のユークリッド距離の計算
# 作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/euclid/"+m.nameFile(main_author,dic,"WordVectorEuclid",str(cate)+"Test",vector,window,epoc,sep,"#","#",".csv"))
outf = open("../result/euclid/"+m.nameFile(main_author,dic,"WordVectorEuclid",str(cate)+"Test",vector,window,epoc,sep,"#","#",".csv"),"a")
for index in T_index_list:
    # ユークリッド距離の計算
    dist = np.sqrt(np.sum(np.square(main_mean-all_works[index].feature)))
    # ファイル書き込み
    print(all_works[index].author,file=outf,end=",")
    print(dist,file=outf,end=",")
    print(all_works[index].filepath,file=outf)
outf.close


