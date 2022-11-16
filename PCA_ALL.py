import sys
import glob
import numpy as np
from gensim.models import word2vec
import Modules as m
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt
import math

'''
実行時引数なし
'''

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
#著者名リスト
author_j_name = {"Akutagawa":"芥川",'Arisima':"有島",'Kajii':"梶井",'Kikuchi':"菊池",'Sakaguchi':"坂口",'Dazai':"太宰",'Nakajima':"中島",'Natsume':"夏目",'Makino':"牧野",'Miyazawa':"宮沢"}
authors_book = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
authors = ['Akutagawa','Sakaguchi','Makino','Dazai','Miyazawa']
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
epoc =300 
#使用モデルのが学習したファイル
model_targets = ["ALL","Learn","ALLCorpus","#"]
model_target = model_targets[2] #ALLCorpus
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
feature_books = np.empty((0, vector), float)
#著者毎の作品名リストを格納するリスト定義
auth_file_list = {}
#全作品のファイル名を保存するリストの初期化
all_file_list = []

##関数定義
def get_id_list(file_obj):
    import csv
    csv_obj = csv.reader(file_obj,delimiter=",")
    id_list = []
    for line in csv_obj:
        id = line[2]
        id_list.append(id)
    return id_list


def get_file_list_none_novel(author):
    #小説リストと作品リストのfileよみこみ
    novel_id_file = open("../data/"+author+"/"+author+"_novellist.csv","r")
    work_id_file = open("../data/"+author+"/"+author+"_worklist.csv","r")
    #idリストの生成
    novel_id_list = get_id_list(novel_id_file)
    work_id_list = get_id_list(work_id_file)
    novel_id_file.close
    work_id_file.close
    #作品idリストから小説作品のidを除外
    for novel_id in novel_id_list:
        try:
            work_id_list.remove(novel_id)
        except:
            print("id:"+novel_id+" is not in "+author+"_worklist")
    #テスト著者の小説以の作品読み込み
    file_list = []
    for work_id in work_id_list:
        file_list.extend(glob.glob('../lcorpus/'+work_id+'_*.txt-utf8-remove-wakati'+dic))
    return file_list


###テスト用作品の特徴抽出--------------------------------------------------------------------------------------

print("do you want use all-authors-model ? ( y or n )")
terminal_input = input()
if( terminal_input == "n" ):
    model_author = ""
    model_type="EACH"
elif( terminal_input == "y" ):
    #全著者モデルの読み込み
    model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
    print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
    model_author = "ALL"
    model_type="ALL"
else:
    print("please input 'y' or 'n'")
    exit()

print("target dic = "+dic)
print(authors)
m.continueOrExit()

#各著者のfeature_booksを作成する
for author in authors:

    print(author)
    if(model_author!="ALL"):
        model = word2vec.Word2Vec.load("../model/"+m.nameFile(author,dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
        print("model = "+m.nameFile(author,dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
        model_author = author

    #対象ファイルをリストアップ
    if(target=="ALL"):
        auth_file_list[author] = glob.glob('../data/'+author+'/*/*.txt-utf8-remove-wakati'+dic)
    elif(target=="ALLCorpus"):
        auth_file_list[author] = glob.glob('../data/'+author+'/*/*.txt-utf8-remove-wakati'+dic)

        none_novel_file_list = get_file_list_none_novel(author)
        auth_file_list[author].extend(none_novel_file_list)
        print(len(auth_file_list[author]))
    else:
        auth_file_list[author] = glob.glob('../data/'+author+'/'+target+'/*.txt-utf8-remove-wakati'+dic)
    #変数初期化
    i=0
    #対象ファイルの読み込み
    for file_name in auth_file_list[author]:
        file = open(file_name)
        data = file.read()

        #ファイル内で使用されている単語を1次元リスト化
        word_list = data.replace("\n"," ").split()
        if len(word_list) == 0:
            print("[WARNING] File("+file_name+") 's word_list length is Zero")

        #作品内で使用されている単語を足し合わせる
        f_ave = np.zeros(vector)
        vocab_vec = np.zeros((vector), float)
        none = 0
        yes = 0
        for word in word_list:
            try:     
                vocab_vec = np.array(model.wv[word]) 
                yes += 1
            except:
                none += 1
                print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
                exit()
            f_ave += vocab_vec

        #print(file_name+" none="+str(none)+" yes="+str(yes))
        #作品の重心（特徴f(A,α)）
        f_ave /= (len(word_list) - none)
        #重心を特徴のリストに保存
        feature_books = np.append(feature_books, np.array([f_ave]), axis=0)
        
        i+=1

        #端末出力
        #print("  "+str(i),end="/")
        #print(len(auth_file_list[author]))


print("PCA")
#主成分分析
pca = PCA(n_components=2)
pca.fit(feature_books)
data_pca = pca.transform(feature_books)

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
    for file_name in auth_file_list[author]:
        x.append(data_pca[num][0])
        y.append(data_pca[num][1])
        num += 1

    plt.scatter(x, y, s=150, marker="o", color=color_list[int(authors_book.index(author))], label=author_j_name[author])
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
#plt.xlim(math.floor(x_min)-1, math.floor(x_max)+1)
#plt.ylim(math.floor(y_min)-1, math.floor(y_max)+1)

title_list={"ALL":"全著者","EACH":"各著者","tLearn":"学習","tTest":"検証","tALL":"全小説","tALLCorpus":"全"}
title_list.update(author_j_name)
title = title_list[model_type]+"モデルから得た各著者の"+title_list["t"+target]+"作品の特徴量分布"
#plt.title(title)
plt.xlabel("第一主成分")
plt.ylabel("第二主成分")

print("save "+m.nameFile("ALL",dic,"pcaGlaph",target,vector,window,epoc,sep,model_target,model_type,".png"))
plt.savefig("../result/pca/"+m.nameFile("ALL",dic,"pcaGlaph",target,vector,window,epoc,sep,model_target,model_type,".png"), bbox_inches='tight')
