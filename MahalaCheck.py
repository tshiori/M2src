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
main_authors = [ 'Akutagawa', 'Sakaguchi', 'Makino' ]
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']

for main_author in main_authors:
    # 著者毎に処理する
    target_f = open("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv"))
    L_file_Names,T_file_Names = m.getFileNameFromTargetListFile(target_f,authors)

    hoge = []
    for T_file_names_auth in T_file_Names.values():
        hoge = hoge + T_file_names_auth   
    T_file_Names = hoge

    file_Names = L_file_Names + T_file_Names

    import numpy as np
    np.random.shuffle(file_Names)
    L_file_Names  = file_Names[0:120]
    T_file_Names = file_Names[120:320]

    #マハラノビス距離の導出に使用したファイルの記録
    import datetime
    import csv
    #書き込みファイルの指定
    m.resetFile("../result/mahala/target/"+m.nameFile(main_author,dic,"RandomList",cate,vector,window,epoc,sep,model_target,other,".csv"))
    outf_target_list = open("../result/mahala/target/"+m.nameFile(main_author,dic,"RandomList",cate,vector,window,epoc,sep,model_target,other,".csv"),"a")
    writer = csv.writer(outf_target_list)
    #現在の日時を記録
    print(datetime.datetime.today(),file=outf_target_list)
    #学習ファイルの記録
    print("main_lerans_index_list",file=outf_target_list)
    writer.writerow(L_file_Names)
    #検証ファイルの記録
    for author in authors:
        print("test_index_list",file=outf_target_list)
        writer.writerow(T_file_Names)
    outf_target_list.close

    #全著者全作品のモデル読み込み
    model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
    print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

    #検証作品の特徴計算############################################################
    #検証作品の重心(特徴)をまとめたリストの初期化
    test_feature_books = np.empty((0, vector), float)
    #検証作品の特徴リストの作成とオブジェクト情報の追加
    for filename in T_file_Names:
        #作品の特徴量f(f_ave)の計算
        f_ave = m.get_f_ave(filename,model,vector)
        #作品の特徴量を特徴のリストに保存
        test_feature_books = np.append(test_feature_books, np.array([f_ave]), axis=0)


    #学習作品の特徴計算############################################################
    #メイン著者学習作品の重心(特徴)をまとめたリストの初期化
    learn_feature_books = np.empty((0, vector), float)
    #学習作品の特徴リストの作成とオブジェクト情報の追加
    for filename in L_file_Names:
        #作品の特徴量f(f_ave)の計算
        f_ave = m.get_f_ave(filename,model,vector)
        #作品の特徴量を特徴のリストに保存
        learn_feature_books = np.append(learn_feature_books, np.array([f_ave]), axis=0)

    #平均(main_mean),の計算とファイル書き込み
    main_mean = np.mean(learn_feature_books, axis=0)


    #分散共分散行列(cov),逆行列(cov_I)の計算
    print(learn_feature_books.shape)
    cov = np.cov(learn_feature_books, rowvar=False,bias=True)
    print("calcurated cov, cov.shape()="+str(cov.shape))
    cov_I = np.linalg.inv(cov)
    print("calcurated cov_I, covI.shape()="+str(cov_I.shape))
    #covとcov_Iのファイル保存


    #距離計算########################################################################
    print("calcurate mahalanobis' distans")
    #作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
    m.resetFile("../result/mahala/random/"+m.nameFile(main_author,dic,"RandomMahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,".csv"))
    mahala_f = open("../result/mahala/random/"+m.nameFile(main_author,dic,"RandomMahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,".csv"),"a")
    for lf in learn_feature_books:
        #偏差(dev),分散共分散行列(cov),逆行列(cov_I)の計算とファイル書き込み
        dev = lf - main_mean
        #計算
        mahala_dist = np.dot(np.dot(dev.T, cov_I), dev)
        #ファイル書き込み
        print(mahala_dist,file=mahala_f)
    mahala_f.close


    #作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
    m.resetFile("../result/mahala/random/"+m.nameFile(main_author,dic,"RandomMahala",str(cate)+"Test",vector,window,epoc,sep,model_target,other,".csv"))
    mahala_f = open("../result/mahala/random/"+m.nameFile(main_author,dic,"RandomMahala",str(cate)+"Test",vector,window,epoc,sep,model_target,other,".csv"),"a")
    for tf in test_feature_books:
        #偏差(dev),分散共分散行列(cov),逆行列(cov_I)の計算とファイル書き込み
        dev = tf - main_mean
        #計算
        mahala_dist = np.dot(np.dot(dev.T, cov_I), dev)
        #ファイル書き込み
        print(mahala_dist,file=mahala_f) 
    mahala_f.close

