import sys
import glob
import numpy as np
from gensim.models import word2vec
import Modules as m
import numpy as np

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
vector = 200
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
#model = word2vec.Word2Vec.load("../model/"+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))
#print("model = "+m.nameFile("ALL",dic,"#","ALL",vector,window,epoc,sep,model_target,other,".model"))

#作品の重心を書き込むファイルの指定と初期化
if(save_ave):
    m.resetFile("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"))
    f = open("../result/average/"+m.nameFile(main_author,dic,"OneHotAve",target,vector,window,epoc,sep,model_target,other,".csv"),"a")


def get_id_list(file_obj):
    import csv
    csv_obj = csv.reader(file_obj,delimiter=",")
    id_list = []
    for line in csv_obj:
        id = line[2]
        id_list.append(id)
    return id_list

def make_one_hot_vector(file_list_all,data=""):
    

    '''
    #小説テキストを入れる変数の初期化
    print("  "+author)
    #[著者名]の[辞書名]のファイルリスト
    file_list = glob.glob('../data/'+author+'/novel/*.txt-utf8-remove-wakati'+dic) 
    print(len(file_list))
    file_list.extend(glob.glob('../data/'+author+'/other/*.txt-utf8-remove-wakati'+dic))
    print(len(file_list))
    '''


    if file_list_all:
        #ある著者の同じ辞書形式の小説文章全てを読み込んで変数に格納
        for file_name in file_list_all:
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
    vocab_list = list(dict.fromkeys(data))
    print("vocab_list_length=",len(vocab_list))   

    #one-hot-vectorの作成
    import sklearn.preprocessing
    word_onehot_dic = {}
    vocabulary_onehot = sklearn.preprocessing.label_binarize(vocab_list,classes=vocab_list)

    for token, onehotvec in zip(vocab_list,vocabulary_onehot):
        word_onehot_dic[token] = onehotvec
        #print("one-hot vector : {}, token : {}".format(onehotvec,token))
    
    return word_onehot_dic,vocab_list
    

def get_f_ave(file_name,word_onehot_dic):
    file = open(file_name)
    data = file.read()

    #ファイル内で使用されている単語を1次元リスト化
    word_list = data.replace("\n"," ").split()
    if len(word_list) == 0:
        print("[WARNING] File("+file_name+") 's word_list length is Zero")

    #作品内で使用されている単語を足し合わせる
    f_ave = np.zeros(len(word_onehot_dic))
    for word in word_list:     
        one_hot_vec = np.array(word_onehot_dic[word])          
        f_ave += one_hot_vec

    #作品の重心（特徴f(A,α)）
    f_ave /= len(word_list)
    return f_ave

def save_ave(author,file_name,f_ave,output_file):
    print(author,file=output_file,end=",")
    f.write(' '.join(list(map(str,f_ave))))
    print(",",file=f,end="")
    print(file_name,file=f)
    

#著者リストの全小説作品リスト内の単語に対するone-hotベクトル辞書の作成
cate1=913
cate2=911
#各著者のインスタンスを生成する
auth_works = {}
index = 0
auth_works,index_fin = m.getWorkObj(main_author,index)

#著者毎の作品名リストを格納するリスト定義
file_list = {}
file_list[cate1] = []
file_list[cate2] = []

for i in range(0,len(auth_works)):
    #重複なし小説リスト
    if(auth_works[i].duplicate==False and auth_works[i].novel==True):
        file_list[cate1] = m.getFileNameByIdUseGlob(main_author,dic,auth_works[i],file_list[cate1])
    #重複なし小説以外の作品リスト
    elif(auth_works[i].duplicate==False and auth_works[i].novel==False and auth_works[i].category==cate2 ):
        file_list[cate2] = m.getFileNameByIdUseGlob(main_author,dic,auth_works[i],file_list[cate2])

#file_list[cate1]とfile_list[cate2]からそれぞれ100作品ずつランダムに選ぶ
import random
for cate in [cate1,cate2]:
    random.seed(42)
    file_list[cate] = random.sample(file_list[cate],100)

file_list_all=file_list[cate1]+file_list[cate2]
print(len(file_list_all))


one_hot_dic,vocab_list = make_one_hot_vector(file_list_all)


#各作品の重心(特徴)をまとめたリストの初期化
feature_books = np.empty((0, len(one_hot_dic)), float)
novel_feature_books = np.empty((0, len(one_hot_dic)), float)
other_feature_books = np.empty((0, len(one_hot_dic)), float)

#小説ファイルの読み込み
for novel_name in file_list[cate1]:
    #作品の特徴量f(f_ave)の計算
    f_ave = get_f_ave(novel_name,one_hot_dic)
    #作品の特徴量を特徴のリストに保存
    novel_feature_books = np.append(novel_feature_books, np.array([f_ave]), axis=0)

#その他のファイルの読み込み
for other_name in file_list[cate2]:
    #作品の特徴量f(f_ave)の計算
    f_ave = get_f_ave(other_name,one_hot_dic)
    #作品の特徴量を特徴のリストに保存
    other_feature_books = np.append(other_feature_books, np.array([f_ave]), axis=0)

feature_books = np.append(novel_feature_books,other_feature_books, axis=0)
#print(len(feature_books))

'''
print("PCA")
#主成分分析
pca = PCA(n_components=2)
pca.fit(feature_books)
data_pca = pca.transform(feature_books)
'''
#print(len(feature_books))
#print(len(feature_books[0]))

#SVM用に整形
print(str(cate1)+" "+str(len(file_list[cate1])))
print(str(cate2)+" "+str(len(file_list[cate2])))
y_novel = [0]*len(file_list[cate1])
y_other = [1]*len(file_list[cate2])
y = np.append(y_novel,y_other, axis=0)
#print(len(y))

#訓練データと検証データに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature_books, y, train_size=int(len(feature_books)*0.7), random_state=0,shuffle=True)

'''
#標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std = X_train
X_test_std = X_test


print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
'''

m.resetFile("../result/svm/nolta/rbf/"+main_author+"SVM-one-hot-C"+other+".csv")
accf=open("../result/svm/nolta/rbf/"+main_author+"SVM-one-hot-C"+other+".csv","a")


for Cnum in [0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0,10,100,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10]:
#for Cnum in [1.0,10,100,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11]:

    from sklearn.svm import SVC
    kernels=["linear","rbf"]
    use_kernel=kernels[1]
    print("parameter C = "+str(Cnum))
    svm = SVC(kernel=use_kernel, C=Cnum, random_state=0)
    svm.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score

    #traning dataのaccuracy
    pred_train = svm.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    print('traning data accuracy： %.2f' % accuracy_train)

    #test data の accuracy
    pred_test = svm.predict(X_test)
    accuracy_test = accuracy_score(y_test, pred_test)
    print('test data accuracy： %.2f' % accuracy_test)

    print(Cnum,",", '%.2f' % accuracy_train,",",'%.2f' %accuracy_test,file=accf)

