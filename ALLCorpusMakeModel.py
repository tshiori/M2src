from gensim.models import word2vec
import glob
import Modules as m
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys


###dataディレクトリの全ての作者と辞書の組み合わせに対してmodelを作成する###
#著者名リスト
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
#辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']

##パラメータ設定==============================
#辞書名
dic = dics[3]
#何のファイルか
what = "#"
#このプログラムの対象
targets = ["ALL","Learn","Test"] 
target = targets[0] #ALL
#学習パラメータ
vector = 100
window = 5
epoc = 500 
mincount = 1
#使用モデルのが学習したファイル
model_targets = ["ALL","Learn"]
model_target = "ALLCorpus"
#その他情報(rowなど)
sep = "#"
other = "alpha-worker1"
#拡張子
extension = ".model"

args = sys.argv
alpha_val=float(args[1])
min_alpha_val=float(args[2])
seed_val=int(args[3])
workers_val=int(args[4])

vector=int(args[5])
# gensim word2vec 初期値
# sentences=None, corpus_file=None, vector_size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, 
# min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>, epochs=5, null_word=0, trim_rule=None,
# sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), comment=None, max_final_vocab=None, shrink_windows=True
##==============================================

data = ""
print(m.nameFile("ALL",dic,"#",target,vector,window,epoc,sep,model_target,other,".model"))
#m.continueOrExit()
import datetime
dt_now = datetime.datetime.now()
log_f=open("../model/model.log","a")
print(dt_now,file=log_f)
print("    model param:",file=log_f)
print("    min_count="+str(mincount)+" window="+str(window)+" size="+str(vector)+" seed="+str(seed_val)+" workers="+str(workers_val)+" alpha="+str(alpha_val)+" min_alpha="+str(min_alpha_val),file=log_f)


for author in authors:
    #小説テキストを入れる変数の初期化
    print("  "+author)
    
    #[著者名]の[辞書名]のファイルリスト
    file_list = glob.glob('../data/'+author+'/novel/*.txt-utf8-remove-wakati'+dic) 
    file_list.extend(glob.glob('../data/'+author+'/other/*.txt-utf8-remove-wakati'+dic))
    print(author+" file num is "+str(len(file_list)))

    if file_list:
        #ある著者の同じ辞書形式の小説文章全てを読み込んで変数に格納
        for file_name in file_list:
            file = open(file_name)
            data += file.read()
        print("    load "+str(len(file_list))+" file is done")

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
    model =  word2vec.Word2Vec( min_count=mincount, window=window, size=vector,seed=seed_val,workers=workers_val,alpha=alpha_val,min_alpha=min_alpha_val )  
    print("build")
    model.build_vocab(data)
    print("build finish")

    #1回学習毎に損失の取得
    i=0
    for i in  tqdm(range(0,epoc)):
        model.train(data, total_examples=model.corpus_count, epochs=1, compute_loss=True)
        loss.append(model.get_latest_training_loss())
        #i += 1
        print(str(loss[i-1]),end=" ")

    print("    made "+dic+" dic model")
    #作成したモデルを"model"ディレクトリへ保存
    model.save("../model/"+m.nameFile("ALL",dic,"#",target,vector,window,epoc,sep,model_target,other,".model"))
    print("    saved model")

    print("誤差保存:")                                                                                                                                                     
    f=open("../result/loss/"+m.nameFile("ALL",dic,"UseModelLoss",target,vector,window,epoc,sep,model_target,other,".txt"),"w")                                                                                                                                                 
    f.write(' '.join(list(map(str,loss))))
    f.close()

    #データの読み込みと数値リスト化
    data = m.readFile("ALL",dic,"UseModelLoss","ALL",vector,window,epoc,sep,model_target,other,".txt","../result/loss/")
    data = data.split()
    data = np.array(list(map(float,data)))

    # 横軸データ生成
    x = np.linspace(0, epoc, epoc)

    print(len(x))
    print(len(data))

    # プロット
    fig = plt.figure()
    ax = plt.axes()
    print(max(data))
    plt.plot(x, data)
    # プロット表示(設定の反映)

    #プロットの保存
    output_file_name = m.nameFile("ALL",dic,"UseModelLoss",target,vector,window,epoc,sep,model_target,other,".png")
    out_dir = "../result/loss/"
    fig.savefig(out_dir+output_file_name)
    print("save : "+out_dir+output_file_name)



print("    "+output_file_name,file=log_f)
log_f.close

print("done")