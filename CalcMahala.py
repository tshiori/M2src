import sys
import numpy as np
from gensim.models import word2vec
import Modules as m

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
targets = ["ALL","Learn","Test"] 
target = targets[0] #ALL
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
dirnum=""
cate=913
#拡張子
extension = ".csv"
##==============================================

##その他初期化
#作品の重心（特徴f(A,α)）の初期化
f_ave = np.zeros(vector)
#Test用ファイルの著者を保存するリストの初期化
author_books = []
#テスト用作品の全ファイル名を保存するリストの初期化
test_list = []

###ターミナル入力----------------------------------------------------------------------------------------------

#著者名リスト
#authors = ['Akutagawa','Sakaguchi','Dazai','Makino','Miyazawa']
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']


#第二引数は推定元の著者名
main_author = sys.argv[1]
if not (main_author in authors):
    print("[ERROR] main_author name is Wrong")
    print(authors)
    exit(1)

###テスト用作品の特徴抽出--------------------------------------------------------------------------------------

#全著者全作品のモデル読み込み
model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))


#作品の重心を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"Ave",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"Ave",target,vector,window,epoc,sep,model_target,other,".csv"),"a")


#各著者のインスタンスを生成する
all_works = {}
index_fin = 0
last_index = 0
#各著者のイインスタンスの生成
for author in authors:
    print("=========================="+author+"===============================")
    #各著者のインスタンスを生成する
    all_works[author],index_fin = m.getWorkObj(author,index_fin)
    print(len(all_works[author]))
    # 確認とインデックス更新
    try:
        print(all_works[author][len(all_works[author])+-1+last_index].index)
        last_index=len(all_works[author])+last_index
    except:
        print("!")
        print(len(all_works[author][len(all_works[author])-1]).id)


#重複なし小説作品のインデックスを著者毎にリスト化する
start=0
nonduplicate_novel_index_list={}
for author in authors:
    nonduplicate_novel_index_list[author]=[]
    for index in range(start,2058):
        try:
            if(all_works[author][index].author != author):
                print(all_works[author][index].author)
            if(all_works[author][index].duplicate==False and all_works[author][index].category==cate):
                nonduplicate_novel_index_list[author].append(all_works[author][index].index)
        except:
            print("fin : "+author)
            start=index
            break
print(all_works[author][index].index)


#nonduplicate_novel_index_list[author]の内容チェック
sum=0
for author in authors:
    #print(len(nonduplicate_novel_index_list[author]))
    sum += len(nonduplicate_novel_index_list[author])
    for index in nonduplicate_novel_index_list[author]:
        if (all_works[author][index].duplicate!=False or all_works[author][index].category!=cate):
            print(all_works[author][index].PrintSelfInformation())
print(sum)


import random
#テスト作品のindexを保存するdictを定義
test_index_list={}
#検証用著者の検証作品20作品をランダムに抽出
for author in authors:
    random.seed(42)
    test_index_list[author] = random.sample(nonduplicate_novel_index_list[author],20)
    test_index_list[author] = sorted(test_index_list[author])


#メイン著者の学習作品120作品を検証作品と重複しないように抽出
import copy
main_learns_index_list=copy.copy(nonduplicate_novel_index_list[main_author])
for l_index in test_index_list[main_author]:
    main_learns_index_list.remove(l_index)
random.seed(42)
main_learns_index_list = random.sample(main_learns_index_list,120)
#学習作品リストを昇順ソートする
main_learns_index_list = sorted(main_learns_index_list)


learns_file_names = []
learns_ids = []
#学習作品のファイルとid抽出
for index in main_learns_index_list:
    learns_file_names.append(all_works[main_author][index].filepath)
    learns_ids.append(all_works[main_author][index].id)


#検証作品のファイル名id抽出
test_file_names = {}
test_ids = {}
for author in authors:
    test_file_names[author]=[]
    test_ids[author]=[]
    for index in test_index_list[author]:
        try:
            test_file_names[author].append(all_works[author][index].filepath)
            test_ids[author].append(all_works[author][index].id)
        except:
            print(author,index)


#マハラノビス距離の導出に使用したファイルの記録
import datetime
import csv
#書き込みファイルの指定
m.resetFile("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,sep,model_target,other,".csv"))
outf_target_list = open("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,sep,model_target,other,".csv"),"a")
writer = csv.writer(outf_target_list)
#現在の日時を記録
print(datetime.datetime.today(),file=outf_target_list)
#学習ファイルの記録
print("main_lerans_index_list",file=outf_target_list)
writer.writerow(learns_file_names)
writer.writerow(learns_ids)
#検証ファイルの記録
for author in authors:
    print("test_index_list["+author+"]",file=outf_target_list)
    writer.writerow(test_file_names[author])
    writer.writerow(test_ids[author])
outf_target_list.close


save_ave=True
#検証作品の特徴計算############################################################
##特徴保存ファイルの設定
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"Ave",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"Ave",target,vector,window,epoc,sep,model_target,other,".csv"),"a")
#検証作品の重心(特徴)をまとめたリストの初期化
test_feature_books = np.empty((0, vector), float)
#検証作品の特徴リストの作成とオブジェクト情報の追加
for author in authors:
   for index in test_index_list[author]:
        #作品の特徴量f(f_ave)の計算
        f_ave = m.get_f_ave(all_works[author][index].filepath,model,vector)
        all_works[author][index].feature = f_ave
        #作品ファイル名と重心を著者辞書Calc.txtファイル書き込み
        if(save_ave):
            print(author,file=f,end=",")
            f.write(' '.join(list(map(str,f_ave))))
            print(",",file=f,end="")
            print(all_works[author][index].filepath,file=f)
        #作品の特徴量を特徴のリストに保存
        test_feature_books = np.append(test_feature_books, np.array([f_ave]), axis=0)
if(save_ave):
    f.close


#学習作品の特徴計算############################################################
##特徴保存ファイルの設定
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"MainAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"MainAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")
#メイン著者学習作品の重心(特徴)をまとめたリストの初期化
learn_feature_books = np.empty((0, vector), float)
#学習作品の特徴リストの作成とオブジェクト情報の追加
for index in main_learns_index_list:
    #作品の特徴量f(f_ave)の計算
    f_ave = m.get_f_ave(all_works[main_author][index].filepath,model,vector)
    all_works[main_author][index].feature = f_ave
    #作品ファイル名と重心を著者辞書Calc.txtファイル書き込み
    if(save_ave):
        print(main_author,file=f,end=",")
        f.write(' '.join(list(map(str,f_ave))))
        print(",",file=f,end="")
        print(all_works[main_author][index].filepath,file=f)
    #作品の特徴量を特徴のリストに保存
    learn_feature_books = np.append(learn_feature_books, np.array([f_ave]), axis=0)
if(save_ave):
    f.close


#平均(main_mean),の計算とファイル書き込み
main_mean = np.mean(learn_feature_books, axis=0)
if(save_ave):
    print("μ=",file=f,end=",")
    f.write(' '.join(list(map(str,main_mean))))
    print("",file=f)
    f.close


#分散共分散行列(cov),逆行列(cov_I)の計算
print(learn_feature_books.shape)
cov = np.cov(learn_feature_books, rowvar=False,bias=True)
print("calcurated cov, cov.shape()="+str(cov.shape))
cov_I = np.linalg.inv(cov)
print("calcurated cov_I, covI.shape()="+str(cov_I.shape))
#covとcov_Iのファイル保存
if(save_cov):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"Cov",target,vector,window,epoc,sep,model_target,other,".csv"))
    covf = open("../result/average/"+m.nameFile(main_author,dic,"Cov",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

    print("cov=",file=covf,end=",")                                          
    covf.write(' '.join(list(map(str,cov))))                                 
    print("",file=covf)                                                      
    print("cov_I=",file=covf,end=",")                                        
    covf.write(' '.join(list(map(str,cov_I))))                               
    print("",file=covf,end="")      

    covf.close



#距離計算########################################################################
print("calcurate mahalanobis' distans")

#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/mahala/"+m.nameFile(main_author,dic,"Mahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,".csv"))
mahala_f = open("../result/mahala/"+m.nameFile(main_author,dic,"Mahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,".csv"),"a")
for index in main_learns_index_list:
    #偏差(dev),分散共分散行列(cov),逆行列(cov_I)の計算とファイル書き込み
    dev = all_works[main_author][index].feature - main_mean
    #計算
    all_works[main_author][index].mahala_dist = np.dot(np.dot(dev.T, cov_I), dev)
    #ファイル書き込み
    print(all_works[main_author][index].author,file=mahala_f,end=",")
    print(all_works[main_author][index].mahala_dist,file=mahala_f,end=",")
    print(all_works[main_author][index].filepath,file=mahala_f,end=",")
    print(all_works[main_author][index].id,file=mahala_f)
mahala_f.close

#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/mahala/"+m.nameFile(main_author,dic,"Mahala",str(cate)+"Test",vector,window,epoc,sep,model_target,other,".csv"))
mahala_f = open("../result/mahala/"+m.nameFile(main_author,dic,"Mahala",str(cate)+"Test",vector,window,epoc,sep,model_target,other,".csv"),"a")
for author in authors:
    for index in test_index_list[author]:
        #偏差(dev),分散共分散行列(cov),逆行列(cov_I)の計算とファイル書き込み
        dev = all_works[author][index].feature - main_mean
        #計算
        all_works[author][index].mahala_dist = np.dot(np.dot(dev.T, cov_I), dev)
        #ファイル書き込み
        print(all_works[author][index].author,file=mahala_f,end=",")
        print(all_works[author][index].mahala_dist,file=mahala_f,end=",")
        print(all_works[author][index].filepath,file=mahala_f,end=",")
        print(all_works[author][index].id,file=mahala_f)
mahala_f.close

