from copy import copy
from ClassWorks import Works
import Modules as m
import csv
import sys
import numpy as np

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



dir_num = ""

##その他初期化
#著者名リスト
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
main_author = sys.argv[1]

#ターゲットファイルを読み出す
targetT_ids = {}
##  result/mahala/targetファイルから学習作品、検証作品のファイル名リストを生成
with open("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv")) as f:
    reader = csv.reader(f)
    next = 0
    ##  学習作品を求める
    for row in reader:
        if(row[0]=='main_lerans_index_list'):
            next = 1
        elif(next==1):
            next = 2
        elif(next==2):
            targetL_ids = row
            next = 0
            break

    ##  検証作品を求める       
    for test_author in authors:
        for row in reader:
            if(row[0]=='test_index_list['+test_author+']'):
                next = 1
            elif(next==1):
                next = 2
            elif(next==2):
                targetT_ids[test_author] = row
                next = 0
                break

#print(targetL_ids)
#print(targetT_ids)
#m.printL(targetL_ids)
#for author in authors:
#    m.printL(targetT_ids[author])


#ターゲットの id:index対応dic
L_IdIndex={}
T_IdIndex={}
#作品インスタンスの生成
works = {}
index = 0
#作品オブジェクトの生成
database_obj = m.openDatabaseCsv(main_author)
for line in database_obj:
    if(line[2] in targetL_ids ):
        works[index] = Works()
        works[index].setSelfInformation(main_author,line,index)
        works[index].setFilePath(dir_num)
        #ターゲットの id:index対応dicを作成
        L_IdIndex[works[index].id]=index
        index=index+1

for test_author in authors:
    database_obj = m.openDatabaseCsv(test_author)
    for line in database_obj:
        if(line[2] in targetT_ids[test_author] ):
            works[index] = Works()
            works[index].setSelfInformation(test_author,line,index)
            works[index].setFilePath(dir_num)
            #ターゲットの id:index対応dicを作成
            T_IdIndex[works[index].id]=index
            index=index+1

#print(L_IdIndex[16])
#print(T_IdIndex)

## 作品距離ファイルの読み出し
# 学習作品のマハラノビス距離結果ファイル読み出し
dataL = m.readFile(main_author,dic,"Mahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,extension,"../result/mahala/")
# ファイル内で使用されている単語を1次元リスト化
mahala_listL = dataL.splitlines()
mahala_listL = [sentence.replace(","," ").split() for sentence in mahala_listL]
# マハラノビス距離の格納
L_mahala_dists = []
for line in mahala_listL:
  if(line[0]==main_author):
    L_mahala_dists.append(float(line[1]))
  else:
    print("main author's learn works mahala dist is not found in...")
    print(m.nameFile(main_author,dic,"Mahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,extension))
    exit()

#ファイル読み出し
dataT = m.readFile(main_author,dic,"Mahala",str(cate)+"Test",vector,window,epoc,sep,model_target,other,extension,"../result/mahala/")
# ファイル内で使用されている単語を1次元リスト化
mahala_listT = dataT.splitlines()
mahala_listT = [sentence.replace(","," ").split() for sentence in mahala_listT]
# マハラノビス距離の格納
for line in mahala_listT:
  if(line[0]==works[T_IdIndex[int(line[3])]].author):
    works[T_IdIndex[int(line[3])]].mahala_dist = float(line[1])
  else:
    print("main author's learn works mahala dist is not found in...")
    print(m.nameFile(main_author,dic,"Mahala",str(cate)+"Learn",vector,window,epoc,sep,model_target,other,extension))
    exit()


#for index in L_IdIndex.values():
#    print(works[index].mahala_dist)
#print("===========================================")
#for index in T_IdIndex.values():
#    print(works[index].mahala_dist)


# しきい値は学習作品のマハラノビス距離のなかで最大のもの
border = max(L_mahala_dists)
#print(border)


# 検証作品のマハラノビス距離が border 以下ならば
import copy
test_authors = copy.copy(authors)
test_authors.remove(main_author)
far_ids = []
for test_author in test_authors:
    for Tid in targetT_ids[test_author]:
        if( works[T_IdIndex[int(Tid)]].mahala_dist  <= border):
            far_ids.append(Tid)

frr_ids = []
for Tid in targetT_ids[main_author]:
    if( works[T_IdIndex[int(Tid)]].mahala_dist > border):
            frr_ids.append(Tid)

# 他人受け入れ率（誤判定率）　＝　メイン著者以外の作品でborderを以下の距離の作品数 / 分類対象者B以外の検証作品数(180)
far = float( float(len(far_ids)) / float(180.0) )


# 本人拒否率　＝　メイン著者の作品でborderを上回る距離の作品数 / 分類対象者Bの検証作品数(20)
frr = float( float(len(frr_ids)) / float(20.0) )


print(far,far_ids)
print(frr,frr_ids)
