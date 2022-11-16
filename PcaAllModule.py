from distutils import filelist
import sys
import glob
import numpy as np
from gensim.models import word2vec
import Modules as m
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt
import inspect
import os


'''
実行時引数なし
'''

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
#著者名リスト
author_j_name = {"Akutagawa":"芥川",'Arisima':"有島",'Kajii':"梶井",'Kikuchi':"菊池",'Sakaguchi':"坂口",'Dazai':"太宰",'Nakajima':"中島",'Natsume':"夏目",'Makino':"牧野",'Miyazawa':"宮沢"}
authors_book = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
#グラフ用カラーリスト
color_list = ['red','hotpink','darkorange','gold','skyblue','green','yellowgreen','brown','blue','black']

##パラメータ設定==============================
#辞書名
dic = dics[3]
#何のファイルか
what = "PCA"
#このプログラムの対象
targets = ["ALL","Learn","Test","ALLCorpus"] 
target = targets[3] 
#学習パラメータ
vector = 100
window = 5 
epoc =500 
#使用モデルのが学習したファイル
#model_targets = ["ALL","Learn","ALLCorpus","#"]
#model_target = model_targets[2] #ALLCorpus
#その他情報(rowなど)
sep = 100
other = "#"
#拡張子
extension = ".csv"


##==============================================

##その他初期化
#カウント用任意変数
i=0


def location(depth=0):
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno


def PCA_ALL_Authors(works,authors,target,vector,model_target,model,category="#"):

    #作品の重心（特徴f(A,α)）の初期化
    f_ave = np.zeros(vector)
    #各作品の重心(特徴)をまとめたリストの初期化
    feature_books = np.empty((0, vector), float)

    #各著者のfeature_booksを作成する
    file_list={}
    for author in authors:
        print(author)
        #対象ファイルをリストアップ
        file_list[author+target]=[]
        if(target=="ALL"):
            dir_target="*"
        elif(target=="novel" or target=="other"):
            dir_target=target

        ##ファイル名を取り出してリスト化
        for i in range(0,len(works)):
            if(works[i].author==author):
                if(category=="#" or works[i].category==int(category)):
                    file_name = glob.glob('../data/'+works[i].author+'/'+dir_target+'/'+str(works[i].id)+'_*.txt-utf8-remove-wakati'+dic)
                    if(len(file_name)!=1):
                        location()
                        print(str(works[i].id))
                        exit()
                    else:
                        file_list[author+target].append(file_name[0])
        
        print("  "+str(len(file_list[author+target])))
    
        #変数初期化
        i=0
        #対象ファイルの本文読み込み
        for file_name in file_list[author+target]:
            file = open(file_name)
            data = file.read()

            #ファイル内で使用されている単語を1次元リスト化
            word_list = data.replace("\n"," ").split()
            if len(word_list) == 0:
                print("[WARNING] File("+file_name+") 's word_list length is Zero")

            #作品内で使用されている単語を足し合わせる
            f_ave = np.zeros(vector)
            vocab_vec = np.zeros((vector), float)
            for word in word_list:
                try:     
                    vocab_vec = np.array(model.wv[word]) 
                except:
                    print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
                    exit()
                f_ave += vocab_vec

            #print(file_name+" none="+str(none)+" yes="+str(yes))
            #作品の重心（特徴f(A,α)）
            f_ave /= (len(word_list))
            #重心を特徴のリストに保存
            feature_books = np.append(feature_books, np.array([f_ave]), axis=0)

        print(len(feature_books))
    print(len(feature_books))
    print(feature_books)

    print("PCA--start")
    #主成分分析
    pca = PCA(n_components=2)
    pca.fit(feature_books)
    data_pca = pca.transform(feature_books)
    print(len(data_pca))
    print("PCA--done")

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
        print("  "+author)
        for file_name in file_list[author+target]:
            x.append(data_pca[num][0])
            y.append(data_pca[num][1])
            num += 1


        plt.scatter(x, y, s=50, marker="o", color=color_list[int(authors_book.index(author))], label=author_j_name[author])
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

    title_list={"ALL":"全","novel":"小説","other":"その他"}
    title_list.update(author_j_name)
    title = "全著者モデルから得た各著者の"+title_list[target]+"作品の特徴量分布"
    #plt.title(title)
    plt.xlabel("第一主成分")
    plt.ylabel("第二主成分")

    png_title=m.nameFile(str(len(authors))+"author",dic,"pcaGlaph",target,vector,window,epoc,sep,model_target,str(category),".png")
    print("save "+png_title)
    plt.savefig("../result/pca/"+png_title, bbox_inches='tight')


###################################################################################################################################

def PCA_Spesific_Authors(works,author,target,vector,model_target,model,category="#"):

    print(author)
    #作品の重心（特徴f(A,α)）の初期化
    f_ave = np.zeros(vector)
    #各作品の重心(特徴)をまとめたリストの初期化
    feature_books = np.empty((0, vector), float)
    #各著者のfeature_booksを作成する
    file_list={}

    #対象ファイルをリストアップ
    file_list[author+target]=[]
    if(target=="ALL"):
        dir_target="*"
    elif(target=="novel" or target=="other"):
        dir_target=target

    ##ファイル名を取り出してリスト化
    for i in range(0,len(works)):
        if(works[i].author==author):
            if(category=="#" or works[i].category==int(category)):
                file_name = glob.glob('../data/'+works[i].author+'/'+dir_target+'/'+str(works[i].id)+'_*.txt-utf8-remove-wakati'+dic)
                if(len(file_name)!=1):
                    location()
                    print(str(works[i].id))
                    exit()
                else:
                    file_list[author+target].append(file_name[0])
    
    print("  "+str(len(file_list[author+target])))

    #変数初期化
    i=0
    #対象ファイルの本文読み込み
    for file_name in file_list[author+target]:
        file = open(file_name)
        data = file.read()

        #ファイル内で使用されている単語を1次元リスト化
        word_list = data.replace("\n"," ").split()
        if len(word_list) == 0:
            print("[WARNING] File("+file_name+") 's word_list length is Zero")

        #作品内で使用されている単語を足し合わせる
        f_ave = np.zeros(vector)
        vocab_vec = np.zeros((vector), float)
        for word in word_list:
            try:     
                vocab_vec = np.array(model.wv[word]) 
            except:
                print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
                exit()
            f_ave += vocab_vec

        #print(file_name+" none="+str(none)+" yes="+str(yes))
        #作品の重心（特徴f(A,α)）
        f_ave /= (len(word_list))
        #重心を特徴のリストに保存
        feature_books = np.append(feature_books, np.array([f_ave]), axis=0)

    print(len(feature_books))

    print("PCA--start")
    #主成分分析
    pca = PCA(n_components=2)
    pca.fit(feature_books)
    data_pca = pca.transform(feature_books)
    print(len(data_pca))
    print("PCA--done")

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

    for file_name in file_list[author+target]:
        x.append(data_pca[num][0])
        y.append(data_pca[num][1])
        num += 1


    #plt.scatter(x, y, s=50, marker="o", color=color_list[int(authors_book.index(author))], label=author_j_name[author])
    plt.scatter(x, y, s=50, marker="o", color="black", label=author_j_name[author])
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

    #plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.subplots_adjust(left=0.05, right=0.7, bottom=0.05, top=0.95)

    title_list={"ALL":"全","novel":"小説","other":"その他"}
    title_list.update(author_j_name)
    title = "全著者モデルから得た各著者の"+title_list[target]+"作品の特徴量分布"
    #plt.title(title)
    plt.xlabel("第一主成分",fontsize=18)
    plt.ylabel("第二主成分",fontsize=18)

    png_title=m.nameFile(author,dic,"pcaGlaph",target,vector,window,epoc,sep,model_target,category,".png")
    print("save "+png_title)
    plt.savefig("../result/pca/"+png_title, bbox_inches='tight')
    plt.clf()

    
    
    
    return 0