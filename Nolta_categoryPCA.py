#ALLCorpusでのPCA分析結果を著者毎に小説（Novel）かそれ以外(other)かに分ける
import numpy as np
from gensim.models import word2vec
import Modules as m
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt

'''
実行時引数なし
'''

###定義と初期化---------------------------------------------------------------
category_forwrd_dict = {911:"poetry",912:"play",913:"novel",914:"essay",915:"diary",916:"record",917:"proverbs",918:"works",919:"chinese writing"}


##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
#著者名リスト
author_j_name = {"Akutagawa":"芥川",'Arisima':"有島",'Kajii':"梶井",'Kikuchi':"菊池",'Sakaguchi':"坂口",'Dazai':"太宰",'Nakajima':"中島",'Natsume':"夏目",'Makino':"牧野",'Miyazawa':"宮沢"}
authors_book = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
authors = ['Akutagawa']
#authors =['Miyazawa']
#グラフ用カラーリスト
color_list = ['red','hotpink','darkorange','gold','skyblue','green','yellowgreen','brown','blue','black']
cate_color = {913:(0.90,0.60,0),914:(0, 0.45, 0.7),911:(0, 0.6, 0.5)} # オレンジ、青、青みの強いみどり

##パラメータ設定==============================
#辞書名
dic = dics[3]
#何のファイルか
what = "PCA"
#このプログラムの対象
targets = ["Novel","Other"] 
target = "" 
#学習パラメータ
vector = 100
window = 5 
epoc =500 
#使用モデルのが学習したファイル
model_targets = ["ALL","Learn","ALLCorpus","#"]
model_target = model_targets[2] #ALLCorpus
#その他情報(rowなど)
sep = "7030"
other = "alpha-worker1"
#拡張子
extension = ".csv"

usemodels = ["each","all","eachCate"]
usemodel = "eachCate"
cate1=913
cate2=914
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


if usemodel == "all":
    #全著者モデルの読み込み
    model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
    print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))


#各著者のインスタンスを生成する
for author in authors:
    print("=========================="+author+"===============================")
    
    if usemodel == "each":
        #各著者モデルの読み込み
        model = word2vec.Word2Vec.load("../model/"+m.nameFile(author,dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
        print( "model = "+m.nameFile(author,dic,"#","ALL",vector,window,epoc,"#",model_target,other,".model"))
    if usemodel == "eachCate":
        #各著者モデルの読み込み
        model_name = m.nameFile(author,dic,"#","200work"+str(cate1)+str(cate2),vector,window,epoc,"#",author+"200work"+str(cate1)+str(cate2),other,".model")
        model = word2vec.Word2Vec.load("../model/"+model_name)
        print( "model = "+model_name)
  

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
        elif(auth_works[i].duplicate==False and auth_works[i].novel==False and auth_works[i].category==cate2 ):
            file_list[cate2] = m.getFileNameByIdUseGlob(author,dic,auth_works[i],file_list[cate2])

    #file_list[cate1]とfile_list[cate2]からそれぞれ100作品ずつランダムに選ぶ
    import random
    for cate in [cate1,cate2]:
        random.seed(42)
        file_list[cate] = random.sample(file_list[cate],100)

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

    for cate in [cate1,cate2]:
        print("  "+author+str(cate))
        for file_name in  file_list[cate]:
            x.append(data_pca[num][0])
            y.append(data_pca[num][1])
            num += 1
        # https://www.nig.ac.jp/color/gen/#color :色盲の人にも色盲でない人にも見やすい色のセット
        if cate==cate1:
            #plt.scatter(x, y, s=300, marker="o", color=color_list[int(authors_book.index(author))], label=category_forwrd_dict[cate1])
            #plt.scatter(x, y, s=300, marker="o", color="black", label=category_forwrd_dict[cate1])
            plt.scatter(x, y, s=300, marker="o", color=cate_color[cate1], label=category_forwrd_dict[cate1]) # オレンジ
        elif cate==cate2:
            #plt.scatter(x, y, s=300, marker="o", color="white", linewidths=1,edgecolors="black",label=category_forwrd_dict[cate2])
            plt.scatter(x, y, s=300, marker="o", color=cate_color[cate2], label=category_forwrd_dict[cate2]) #青みの強い緑　宮沢賢治詩歌
            #plt.scatter(x, y, s=300, marker="o", color=(0, 0.45, 0.7), label=category_forwrd_dict[cate2]) #青 芥川龍之介評論
        else:
            print("ERROR : category "+cate+" is invalid value")

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

    plt.legend(fontsize=60,bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.subplots_adjust(left=0.05, right=0.7, bottom=0.05, top=0.95)
    #plt.xlim(math.floor(x_min), math.floor(x_max))
    #plt.ylim(math.floor(y_min), math.floor(y_max))

    title = author+"モデルから得た"+author_j_name[author]+"の全作品の特徴量分布"
    #plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    #model_target = author+"ALLCorpus"
    if usemodel == "all":
        print("save "+m.nameFile(author,dic,"pcaGlaphNO","ALLCorpus",vector,window,epoc,sep,model_target,"color",".png"))
        plt.savefig("../result/pca/Nolta/"+m.nameFile(author,dic,"pcaGlaphNO","ALLCorpus",vector,window,epoc,sep,model_target,"color",".png"), bbox_inches='tight')
    elif usemodel == "each":
        print("save "+m.nameFile(author,dic,"pcaGlaphNO","ALLCorpus",vector,window,epoc,sep,author,"color",".png"))
        plt.savefig("../result/pca/Nolta/"+m.nameFile(author,dic,"pcaGlaphNO","ALLCorpus",vector,window,epoc,sep,author,"color",".png"), bbox_inches='tight')
    elif usemodel == "eachCate":
        print("save "+m.nameFile(author,dic,"pcaGlaphNO","200work"+str(cate1)+str(cate2),vector,window,epoc,sep,author+"200work"+str(cate1)+str(cate2),"color",".png"))
        plt.savefig("../result/pca/Nolta/"+m.nameFile(author,dic,"pcaGlaphNO","200work"+str(cate1)+str(cate2),vector,window,epoc,sep,author+"200work"+str(cate1)+str(cate2),"color",".png"), bbox_inches='tight')
    plt.clf()
    plt.cla()
