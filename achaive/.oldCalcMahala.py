import sys
import glob
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
save_ave = False #False->出力しない True->出力する
save_cov = False

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
epoc = 300 
#使用モデルのが学習したファイル
model_targets = ["*","ALL","Learn","ALLCorpus"]
model_target = model_targets[3] # *
#その他情報(rowなど)
sep = 100
other = "#"
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

###テスト用作品の特徴抽出--------------------------------------------------------------------------------------

#全著者全作品のモデル読み込み
model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

#作品の重心を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"Ave",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"Ave",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

#全著者の検証用ファイルを
for test_author in authors:
    print(test_author)

    if(target=="ALL"):
        #targetをワイルドカードとして対象のテスト用ファイルをリストアップ
        file_list = glob.glob('../data/'+test_author+'/*/*.txt-utf8-remove-wakati'+dic)
    else:
        #対象のテスト用ファイルをリストアップ
        file_list = glob.glob('../data/'+test_author+'/'+target+'/*.txt-utf8-remove-wakati'+dic)

    i=0
    #対象ファイルの読み込み
    for file_name in file_list:
        file = open(file_name)
        data = file.read()

        #ファイル内で使用されている単語を1次元リスト化
        word_list = data.replace("\n"," ").split()
        if len(word_list) == 0:
            print("[WARNING] File("+file_name+") 's word_list length is Zero")

        #作品内で使用されている単語を足し合わせる
        f_ave = np.zeros(vector)
        for word in word_list:     
            vocab_vec = np.array(model.wv[word])          
            if 0 in vocab_vec:
                print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
            f_ave += vocab_vec

        #作品の重心（特徴f(A,α)）
        f_ave /= len(word_list)
        #重心を特徴のリストに保存
        test_feature_books = np.append(test_feature_books, np.array([f_ave]), axis=0)
        
        #作品ファイル名と重心を著者辞書Calc.txtファイル書き込み
        if(save_ave):
            print(test_author,file=f,end=",")
            f.write(' '.join(list(map(str,f_ave))))
            print(",",file=f,end="")
            print(file_name,file=f)
        
        i+=1

        #端末出力
        '''
        print("  "+str(i),end="/")
        print(len(file_list))
        '''

        author_books.append(test_author)

    test_list = test_list + file_list
    print(len(test_list))

if(save_ave):
    f.close


###推定元作者の特徴μとcov_Iの導出--------------------------------------------------------------------------------------

#作品の重心と著者の特徴を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"MainAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"MainAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

i=0
if(target=="ALL"):
    #メイン著者の全ての作品を学習データとしてリストアップ
    learn_list = glob.glob('../data/'+main_author+'/*/*.txt-utf8-remove-wakati'+dic)
else:
    #対象著者の学習データ用ファイルをリストアップ
    learn_list = glob.glob('../data/'+main_author+'/Learn/*.txt-utf8-remove-wakati'+dic)

#著者の特徴main_meanの初期化
main_mean = np.zeros(vector)
auth_word_num = 0
#対象ファイルの読み込み
for file_name in learn_list:
    file = open(file_name)
    data = file.read()
    #ファイル内で使用されている単語を1次元リスト化
    word_list = data.replace("\n"," ").split()
    if len(word_list) == 0:
            print("[WARNING] File("+file_name+") 's word_list length is Zero")
    f_ave = np.zeros(vector)            
    for word in word_list:
        #作品の特徴（平均）計算
        vocab_vec = np.array(model.wv[word])
        if 0 in vocab_vec:
            print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
        f_ave += vocab_vec

    #作品の重心（特徴f(A,α)）
    f_ave /= len(word_list)
    #重心を特徴のリストに保存
    main_feature_books = np.append(main_feature_books, np.array([f_ave]), axis=0)
    #作品の単語数を足し合わせて著者の全学習作品中の単語数を保存する
    auth_word_num += len(word_list)
    if(save_ave):
        print(main_author,file=f,end=",")
        f.write(' '.join(list(map(str,f_ave))))
        print(",",file=f,end="")
        print(file_name,file=f)
    '''    
    #ターミナル出力
    i+=1
    print("  "+str(i),end="/")
    print(len(learn_list))
    '''

#平均(main_mean),の計算とファイル書き込み
main_mean = np.mean(main_feature_books, axis=0)

if(save_ave):
    print("μ=",file=f,end=",")
    f.write(' '.join(list(map(str,main_mean))))
    print("",file=f)
    f.close


#分散共分散行列(cov),逆行列(cov_I)の計算
print(main_feature_books.shape)
cov = np.cov(main_feature_books, rowvar=False,bias=True)
print("calcurated cov, cov.shape()="+str(cov.shape))
cov_I = np.linalg.inv(cov)
print("calcurated cov_I, covI.shape()="+str(cov_I.shape))


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

###テスト作品のマハラノビス距離の導出------------------------------------------------------------------------------------

#偏差(dev),分散共分散行列(cov),逆行列(cov_I)の計算とファイル書き込み
dev = test_feature_books - main_mean


#距離計算(main_dist)
print("calcurate mahalanobis' distans")

test_dist = np.empty(len(test_list), float)

#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/mahala/"+m.nameFile(main_author,dic,"Mahala",target,vector,window,epoc,sep,model_target,other,".csv"))
mahala_f = open("../result/mahala/"+m.nameFile(main_author,dic,"Mahala",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

i=0
print("test_list length = "+str(len(test_list)))
print(len(author_books))
#作品ファイル毎にマハラノビス距離を計算
for file_name in test_list:
    #計算
    test_dist[i] = np.dot(np.dot(dev[i].T, cov_I), dev[i])
    #ファイル書き込み
    print(author_books[i],file=mahala_f,end=",")
    print(test_dist[i],file=mahala_f,end=",")
    print(file_name,file=mahala_f)
    i+=1
    #ターミナル出力
    #print("  "+str(i),end="/")
    #print(len(test_list))
