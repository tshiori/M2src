import sys
from gensim.models import word2vec
import Modules as m
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

###定義と初期化---------------------------------------------------------------

##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
#著者名リスト
author_j_name = {"Akutagawa":"芥川",'Arisima':"有島",'Kajii':"梶井",'Kikuchi':"菊池",'Sakaguchi':"坂口",'Dazai':"太宰",'Nakajima':"中島",'Natsume':"夏目",'Makino':"牧野",'Miyazawa':"宮沢"}
#authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
authors = ['Akutagawa']
authors = ['Miyazawa']
#authors = ['Sakaguchi']
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
vector = 80
window = 5 
epoc =500
mincount = 1
m_seed_val=1
workers_val=1
min_alpha_val=0.0001 #0.0001がデフォルト
alpha_val=0.025 # 0.025がデフォルト
#使用モデルのが学習したファイル
model_targets = ["ALL","Learn","ALLCorpus","#"]
model_target = ""
#その他情報(rowなど)
sep = "7030"
other = "alpha-worker1-2"
#拡張子
extension = ".csv"

dirnum=""
usemodels = ["each","all"]
usemodel = "each"
cate1=913
cate2=914

##==============================================


#各著者のインスタンスを生成する
main_author = ""
try:
    main_author = sys.argv[1]
except:
    main_author = input("input main author name = ")
print("main_author = "+main_author)

try:
    cate2=int(sys.argv[2])
    print(cate2)
except:
    exit()
try:
    other=sys.argv[3]
    print(other)
except:
    exit()
try:
    dirnum=sys.argv[4]
    print(dirnum)
except:
    exit()

#各著者モデルの読み込み
model_name = m.nameFile(main_author,dic,"#","200work"+str(cate1)+str(cate2),vector,window,epoc,"#",main_author+"200work"+str(cate1)+str(cate2),other,".model")
print(model_name)

#m.continueOrExit()
import datetime
dt_now = datetime.datetime.now()
log_f=open("../model/model.log","a")
print(dt_now,file=log_f)
print("    model param:",file=log_f)
print("    min_count="+str(mincount)+" window="+str(window)+" size="+str(vector)+" seed="+str(m_seed_val)+" workers="+str(workers_val)+" alpha="+str(alpha_val)+" min_alpha="+str(min_alpha_val),file=log_f)
print("    "+model_name,file=log_f)
print("=========================="+main_author+"===============================")
#各著者のインスタンスを生成する
auth_works = {}
index = 0
auth_works,index_fin = m.getWorkObj(main_author,index)

#著者毎の作品名リストを格納するリスト定義
file_list = {}
file_list[cate1] = []
file_list[cate2] = []
print(cate1,cate2)
for i in range(0,len(auth_works)):
    #重複なし小説リスト
    if(auth_works[i].duplicate==False and auth_works[i].novel==True):
        file_list[cate1] = m.getFileNameByIdUseGlob(main_author,dic,auth_works[i],file_list[cate1],dirnum)
    #重複なし小説以外の作品リスト
    elif(auth_works[i].duplicate==False and auth_works[i].novel==False and auth_works[i].category==cate2 ):
        file_list[cate2] = m.getFileNameByIdUseGlob(main_author,dic,auth_works[i],file_list[cate2],dirnum)
print(len(file_list[cate1]))
print(len(file_list[cate2]))
exit()
#file_list[cate1]とfile_list[cate2]からそれぞれ100作品ずつランダムに選ぶ
import random
for cate in [cate1,cate2]:
    random.seed(42)
    m.printL(file_list[cate])
    file_list[cate] = random.sample(file_list[cate],100)
'''
debugf = open("../debug/"+main_author+"_filecheck_model.txt","w")
print(file_list[cate1],file=debugf)
print(file_list[cate2],file=debugf)
debugf.close
'''
data = ""
for cate in [cate1,cate2]:
    if file_list[cate]:
        #ある著者の同じ辞書形式の小説文章全てを読み込んで変数に格納
        for file_name in file_list[cate]:
            file = open(file_name)
            data += file.read()
        print("    load "+str(len(file_list[cate]))+" file is done")

data = data.replace("\n"," ")
data = data.split()
m.printL(data)

if data!="":
    if(other == "1row"):
        print("1row")
        #小説文章を1次元リスト構造に整形
        data = data.replace("\n"," ")
        data = data.split()
    else:
        #小説文章を二重のリスト構造に整形
        data = data.splitlines()
        data = [sentence.split() for sentence in data]

    '''
    #小説文章を学習してモデルを作成
    model =  word2vec.Word2Vec( data, min_count=1, window=window, iter=epoc, size=vector)
    '''

    #小説文章を初回学習してモデルを作成
    loss = []
    model =  word2vec.Word2Vec( min_count=mincount, window=window, size=vector,seed=m_seed_val,workers=workers_val,alpha=alpha_val,min_alpha=min_alpha_val )  
    print("build")
    model.build_vocab(data)
    print("build finish")

    #1回学習毎に損失の取得
    i=0
    for i in tqdm(range(0,epoc)):
        model.train(data, total_examples=model.corpus_count, epochs=1, compute_loss=True)
        loss.append(model.get_latest_training_loss())


    print("    made "+dic+" dic model")
    #作成したモデルを"model"ディレクトリへ保存
    model.save("../model/"+model_name)
    print("    saved model")

    print("誤差保存:")
    f=open("../result/loss/"+model_name[:-5]+"txt","w") 
    f.write(' '.join(list(map(str,loss))))
    f.close()

    #データの読み込みと数値リスト化
    '''
    dirpass は「/」含める。ex.) "../result/mahala/"
    '''
    print("readFile : "+"../result/loss/"+model_name[:-5]+"txt")
    file = open("../result/loss/"+model_name[:-5]+"txt")
    data = file.read()

    data = data.split()
    data = np.array(list(map(float,data)))

    # 横軸データ生成
    x = np.linspace(0, epoc, epoc)

    # プロット
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(x, data)
    # プロット表示(設定の反映)

    #プロットの保存
    output_file_name = model_name[:-5]+"png"
    out_dir = "../result/loss/"
    fig.savefig(out_dir+output_file_name)
    print("save : "+out_dir+output_file_name)
else:
    print("data == null")

print("done")

log_f.close
