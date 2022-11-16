#ALLCorpusでのPCA分析結果を著者毎に小説（Novel）かそれ以外(other)かに分ける
import sys
import glob
import numpy as np
from gensim.models import word2vec
import Modules as m
import sklearn #機械学習のライブラリ
#rom sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt
import math
from ClassWorks import Works
import inspect
import os
import csv
import PcaAllModule as pa


'''
実行時引数なし
'''

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
#著者名リスト
author_j_name = {"Akutagawa":"芥川",'Arisima':"有島",'Kajii':"梶井",'Kikuchi':"菊池",'Sakaguchi':"坂口",'Dazai':"太宰",'Nakajima':"中島",'Natsume':"夏目",'Makino':"牧野",'Miyazawa':"宮沢"}
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
#authors = ['Akutagawa','Sakaguchi','Makino','Dazai','Miyazawa']
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
other = "alpha-worker1"
#拡張子
extension = ".csv"

args = sys.argv
vector=int(args[1])
##==============================================

##その他初期化
#カウント用任意変数
i=0
#作品の重心（特徴f(A,α)）の初期化
f_ave = np.zeros(vector)
#各作品の重心(特徴)をまとめたリストの初期化
feature_books = np.empty((0, vector), float)
#著者毎の作品名リストを格納するリスト定義
file_list = {}
#全作品のファイル名を保存するリストの初期化
all_file_list = []


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


###テスト用作品の特徴抽出--------------------------------------------------------------------------------------


#全著者モデルの読み込み
print(authors)
print("target dic = "+dic)
m.continueOrExit()



def openDatabaseCsv(author):
    #database.csvの読み込み
    database = open("../data/"+author+"/"+author+"_database.csv","r")
    csv_obj = csv.reader(database,delimiter=",")
    database.close
    return csv_obj


def getWorkObj(author,index_0):
    database_obj = openDatabaseCsv(author)
    #作品インスタンスの生成
    works = {}
    index = index_0
    for line in database_obj:
        works[index] = Works()
        works[index].setSelfInformation(author,line,index)
        works[index].setFilePath()
        index=index+1
    return works,index

def location(depth=0):
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno


#全著者全作品（重複あり）のインスタンスを生成する
all_works = {}
index_fin = 0
#past = 0
for author in authors:
    print("################"+author+"###############")
    all_works[author],index_fin = getWorkObj(author,index_fin)
    print(index_fin)


#ターゲットのインスタンスだけにする
j=0
k=0
target_works={}
for author in authors:
    print(author+" : "+str(len(all_works[author])))
    for i in range(0,len(all_works[author])):
        try:
            id = all_works[author][j].id
        except:
           print(author,j)

        #重複ファイルかどうか
        if(all_works[author][j].duplicate==True):
            pass
        elif(all_works[author][j].duplicate==False):
            #ターゲットかどうか
            if(target=="novel" and all_works[author][j].novel==True):
                target_works[k]=all_works[author][j]
                k=k+1
            elif(target=="other" and all_works[author][j].novel==False):
                target_works[k]=all_works[author][j]
                k=k+1
            elif(target=="ALL"):
                target_works[k]=all_works[author][j]
                k=k+1
            else:
                print(target)
        else:
            print(location())
            print(author+" id:"+str(all_works[j].id)+" works.duplicate is void")
        j=j+1


###全著者モデルから取り出した
#全著者モデルの読み込み
model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

#authorsで渡した著者の作品のPCA結果を1枚の画像で出力する
pa.PCA_ALL_Authors(target_works,authors,target,vector,model_target+"-"+other,model,913)


#authorsで渡した著者の作品のPCA結果を各著者について画像で出力する
for author in authors:
    pa.PCA_Spesific_Authors(target_works,author,target,vector,model_target+"-"+other,model,913)
    print("PCA_Spesific_Authors--from all authors model done")

#authorsで渡した著者のみのモデルを使って作品のPCA結果を各著者について画像で出力する
for author in authors:
    model = word2vec.Word2Vec.load("../model/"+m.nameFile(author,dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
    print("model = "+m.nameFile(author,dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
    pa.PCA_Spesific_Authors(target_works,author,target,vector,author+model_target+"-"+other,model,913)
    print("PCA_Spesific_Authors--from "+author+" model done")

print("PCA.py all done")

exit()







#各著者のfeature_booksを作成する
file_list = {}
for author in authors:
    #print("model = "+m.nameFile(author,dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
    #model = word2vec.Word2Vec.load("../model/"+m.nameFile(author,dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
    print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
    model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))



    print(author)

    #小説ファイルをリストアップ
    file_list[author+"Novel"] = glob.glob('../data/'+author+'/*/*.txt-utf8-remove-wakati'+dic)
    #対象ファイルの読み込み
    for novel_name in file_list[author+"Novel"]:
        #作品の特徴量f(f_ave)の計算
        f_ave = get_f_ave(novel_name)
        #作品の特徴量を特徴のリストに保存
        feature_books = np.append(feature_books, np.array([f_ave]), axis=0)

    #小説以外のファイルをリストアップ
    file_list[author+"Other"] = get_file_list_none_novel(author)
    #対象ファイルの読み込み
    for other_name in file_list[author+"Other"]:
        #作品の特徴量f(f_ave)の計算
        f_ave = get_f_ave(other_name)
        #作品の特徴量を特徴のリストに保存
        feature_books = np.append(feature_books, np.array([f_ave]), axis=0)



print("PCA")
#主成分分析
pca = PCA(n_components=2)
pca.fit(feature_books)
data_pca = pca.transform(feature_books)

# 主成分の寄与率を出力
print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
print("exit")

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

for author in authors:
    for target in targets:
        print("  "+author+target)
        for file_name in file_list[author+target]:
            x.append(data_pca[num][0])
            y.append(data_pca[num][1])
            num += 1

        if target=="Novel":
            plt.scatter(x, y, s=300, marker="o", color=color_list[int(authors_book.index(author))], label=author_j_name[author]+"小説")
        elif target=="Other":
            plt.scatter(x, y, s=500, marker="*", color="white", linewidths=1,edgecolors=color_list[int(authors_book.index(author))],label=author_j_name[author]+"小説以外")
        else:
            print("ERROR : target "+target+" is invalid value")

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

    title = author+"モデルから得た"+author_j_name[author]+"の全作品の特徴量分布"
    #plt.title(title)
    plt.xlabel("第一主成分")
    plt.ylabel("第二主成分")
    #model_target = author+"ALLCorpus"
    print("save "+m.nameFile(author,dic,"pcaGlaphNO","ALLCorpus",vector,window,epoc,"#",model_target,other,".png"))
    plt.savefig("../result/pca/"+m.nameFile(author,dic,"pcaGlaphNO","ALLCorpus",vector,window,epoc,"#",model_target,other,".png"), bbox_inches='tight')
    plt.clf()
    plt.cla()
