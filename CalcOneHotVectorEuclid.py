import sys
import glob
import numpy as np
import Modules as m
import numpy as np
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
target = targets[3] #ALLCorpus
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
#model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
#print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

#作品の重心を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")

  

#著者リストの全小説作品リスト内の単語に対するone-hotベクトル辞書の作成
cate=913
#各著者のインスタンスを生成する
auth_works = {}
index = 0
auth_works,index_fin = m.getWorkObj(main_author,index)


#ターゲットのファイル名リスト生成
f = open("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv"))
L_file_names,T_file_names = m.getFileNameFromTargetListFile(f,authors)
f.close


## Word2Vecで学習したものと同じ全ファイル名を取得
# 小説テキストを入れる変数の初期化
data = ""
for author in authors:  
    #[著者名]の[辞書名]のファイルリスト
    all_file_list = glob.glob('../data/'+author+'/novel/*.txt-utf8-remove-wakati'+dic) 
    all_file_list.extend(glob.glob('../data/'+author+'/other/*.txt-utf8-remove-wakati'+dic))
    print(author+" file num is "+str(len(all_file_list)))

    #ファイル名のリストをシャッフル
    '''
    import random
    random.seed(42)
    random.shuffle(all_file_list)
    '''

    if all_file_list:
        #ある著者の同じ辞書形式の小説文章全てを読み込んで変数に格納
        for file_name in all_file_list:
            file = open(file_name)
            data += file.read()
        #print("    load "+str(len(all_file_list))+" file is done")


## 全語彙リストの作成
if all_file_list:
    #ある著者の同じ辞書形式の小説文章全てを読み込んで変数に格納
    for file_name in all_file_list:
        file = open(file_name)
        data += file.read()
    print("    load file is done")

if data!="":
    #小説文章を1次元リスト構造に整形
    data = data.replace("\n"," ")
    data = data.split()
else:
    print("[ERROR] none data")
    exit(1)

#単語の重複を許さない語彙のリスト 
#print("data_length=",len(data))
vocab_list = list(dict.fromkeys(data))
print("vocab_list_length=",len(vocab_list))   
#m.printSizeUnit(data,"M")
#m.printSizeUnit(vocab_list,"M")
#m.printSizeUnit(all_file_list,"k")
del data

## one-hot-vector表現での学習作品の特徴量から著者の特徴量を求める
# メイン著者の特徴量の定義
print("Calc main author's feature")
auth_f = np.zeros(len(vocab_list),dtype=float)
from tqdm import tqdm
for L_file_name in tqdm(L_file_names):
    # 単語の出現頻度を集計するdic
    count_vocab = {}
    file = open(L_file_name)
    data = file.read()

    # ファイル内で使用されている単語を1次元リスト化
    word_list = data.replace("\n"," ").split()
    if len(word_list) == 0:
        print("[WARNING] File("+L_file_name+") 's word_list length is Zero")

    # 作品内で使用されている単語の出現頻度を集計
    for word in word_list:
        try:
            count_vocab[vocab_list.index(word)] += 1.0
        except:
            count_vocab[vocab_list.index(word)] = 1.0
        # print(type(count_vocab[vocab_list.index(word)]),count_vocab[vocab_list.index(word)])

    # 作品の語彙数とcount_vocabの長さが異なったらエラー
    if(len(list(set(word_list)))!=len(count_vocab)):
        print("Can not count frequency of appearance of words correctly in file:"+L_file_name)
        print("count_vocab length : "+str(len(count_vocab)),"!= vocablary length :"+str(len(list(set(word_list)))))
        exit()
    
    # One-hot ベクトル表現での特徴量生成
    f_onehot_ave = np.zeros(len(vocab_list),dtype=float)
    for i in range(0,len(vocab_list)):
        try:
            f_onehot_ave[i] = count_vocab[i]
        except:
            pass
    f_onehot_ave /= len(count_vocab)
    auth_f += f_onehot_ave
    file.close
# メイン著者の特徴量
auth_f/=len(L_file_names)

## 学習作品のユークリッド距離の計算
#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/euclid/"+m.nameFile(main_author,dic,"OneHotVectorEuclid",str(cate)+"Learn",vector,window,epoc,sep,"#","#",".csv"))
outf = open("../result/euclid/"+m.nameFile(main_author,dic,"OneHotVectorEuclid",str(cate)+"Learn",vector,window,epoc,sep,"#","#",".csv"),"a")
# メイン著者の特徴量の定義
print("Calc learn file's euclid distance and save file...")
for L_file_name in tqdm(L_file_names):
    # 単語の出現頻度を集計するdic
    count_vocab = {}
    file = open(L_file_name)
    data = file.read()
    file.close

    # ファイル内で使用されている単語を1次元リスト化
    word_list = data.replace("\n"," ").split()
    if len(word_list) == 0:
        print("[WARNING] File("+L_file_name+") 's word_list length is Zero")

    # 作品内で使用されている単語の出現頻度を集計
    for word in word_list:
        try:
            count_vocab[vocab_list.index(word)] += 1.0
        except:
            count_vocab[vocab_list.index(word)] = 1.0
        # print(type(count_vocab[vocab_list.index(word)]),count_vocab[vocab_list.index(word)])

    # 作品の語彙数とcount_vocabの長さが異なったらエラー
    if(len(list(set(word_list)))!=len(count_vocab)):
        print("Can not count frequency of appearance of words correctly in file:"+L_file_name)
        print("count_vocab length : "+str(len(count_vocab)),"!= vocablary length :"+str(len(list(set(word_list)))))
        exit()
    
    # One-hot ベクトル表現での特徴量生成
    f_onehot_ave = np.zeros(len(vocab_list),dtype=float)
    for i in range(0,len(vocab_list)):
        try:
            f_onehot_ave[i] = count_vocab[i]
        except:
            pass
    f_onehot_ave /= len(count_vocab)

    # ユークリッド距離の計算
    dist = np.sqrt(np.sum(np.square(auth_f-f_onehot_ave)))
    # ファイル書き込み
    print(main_author,file=outf,end=",")
    print(dist,file=outf,end=",")
    print(L_file_name,file=outf)    
outf.close

# 検証作品のユークリッド距離の計算
#作品ファイル名とそのマハラノビス距離を書き込むファイルの指定と初期化
m.resetFile("../result/euclid/"+m.nameFile(main_author,dic,"OneHotVectorEuclid",str(cate)+"Test",vector,window,epoc,sep,"#","#",".csv"))
outf = open("../result/euclid/"+m.nameFile(main_author,dic,"OneHotVectorEuclid",str(cate)+"Test",vector,window,epoc,sep,"#","#",".csv"),"a")
# メイン著者の特徴量の定義
print("Calc test file's euclid distance and save file...")
for author in authors:
    for T_file_name in tqdm(T_file_names[author]):
        # 単語の出現頻度を集計するdic
        count_vocab = {}
        file = open(T_file_name)
        data = file.read()
        file.close

        # ファイル内で使用されている単語を1次元リスト化
        word_list = data.replace("\n"," ").split()
        if len(word_list) == 0:
            print("[WARNING] File("+T_file_name+") 's word_list length is Zero")

        # 作品内で使用されている単語の出現頻度を集計
        for word in word_list:
            try:
                count_vocab[vocab_list.index(word)] += 1.0
            except:
                count_vocab[vocab_list.index(word)] = 1.0
            # print(type(count_vocab[vocab_list.index(word)]),count_vocab[vocab_list.index(word)])

        # 作品の語彙数とcount_vocabの長さが異なったらエラー
        if(len(list(set(word_list)))!=len(count_vocab)):
            print("Can not count frequency of appearance of words correctly in file:"+T_file_name)
            print("count_vocab length : "+str(len(count_vocab)),"!= vocablary length :"+str(len(list(set(word_list)))))
            exit()
        
        # One-hot ベクトル表現での特徴量生成
        f_onehot_ave = np.zeros(len(vocab_list),dtype=float)
        for i in range(0,len(vocab_list)):
            try:
                f_onehot_ave[i] = count_vocab[i]
            except:
                pass
        f_onehot_ave /= len(count_vocab)

        # ユークリッド距離の計算
        dist = np.sqrt(np.sum(np.square(auth_f-f_onehot_ave)))
        # ファイル書き込み
        print(author,file=outf,end=",")
        print(dist,file=outf,end=",")
        print(T_file_name,file=outf)
        
outf.close

exit()
