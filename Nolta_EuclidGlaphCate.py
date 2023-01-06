'''
ターゲットが"Learn"のとき
メイン著者の120作品の学習作品のみを描画
ターゲットが"Test"のとき
メイン著者を含むauthorsに指定した全著者の検証作品のマハラノビス距離を描画
'''
#@title デフォルトのタイトル テキスト
import numpy as np
import matplotlib.pyplot as plt
import sys
import Modules as m

#著者名リスト   0          1        2        3         4          5        6         7         8         9
allauthors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
#authors = ['Akutagawa','Sakaguchi','Dazai','Makino','Miyazawa']
#authors = ["Sakaguchi"]
auth_num = str(len(authors))
#グラフの色　　0        1           2         3       4       5           6          7        8      9
color_list = ['red','hotpink','darkorange','gold','skyblue','green','yellowgreen','brown','blue','black'] #Testのとき
#color_list = ['red','skyblue','green','blue','black'] #Lernのとき
#辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']

##パラメータ-------------------------------------------
#ファイル情報
dic = "uNeologd"
main_author = ""
#what = "OneHotVectorEuclid"
#what = "GenreEuclid"
what = "GenreMahala"
#what = "WordVectorEuclid"
targets = ["ALL","Learn","Test","ALLCorpus","NoneNovel","Novel"]  
target = ""
#モデルの学習パラメータ
'''
#Onehot vector
vector = "#"
window = "#"
epoc = "#"
'''

#Word vector vector
vector = 100
window = 5
epoc = 500


cate1=913
cate2=914

#使用モデルのが学習したファイル
model_targets = ["ALL","Learn","ALLCorpus","#"]
model_target = "ALLCorpus"
#コーパス分割割合，その他
sep = 100
other = "#"
#拡張子
extension =".csv"

#グラフの右端マハラノビス距離
histmax = 0
#-----------------------------------------------------

main_author = sys.argv[1]
if not (main_author in authors):
    print("[ERROR] author name is Wrong")
    print(authors)
    exit(1)
try:
    cate2=int(sys.argv[2])
    print(cate2)
except:
    exit()
try:
    histmax=int(sys.argv[3])
    print(histmax)
except:
    exit()
try:
    order=int(sys.argv[4])
    print(order)
except:
    exit()

'''
### 入力例　mainauthor="Akutagawa" cate="913" histmax="10000"
main_author = input("input main author name =  ")
#main_author = "Akutagawa"
#cate = input("input category (NDC) =  ")
target = "Test"
histmax = int(input("input histmax =  "))
'''
output_file_name = m.nameFile(main_author,dic,what+"Glaph",str(cate1)+str(cate2),vector,window,epoc,sep,model_target,"histmax"+str(histmax),".png")
print(output_file_name)
m.continueOrExit()

#ファイル読み出し
data = m.readFile(main_author,dic,what,str(cate1)+str(cate2),vector,window,epoc,sep,model_target,other,extension,"../result/euclid/nolta/")
#data = m.readFile(main_author,dic,"OneHotVectorEuclid",target,vector,window,epoc,sep,model_target,other,extension,"../result/euclid/")

#ファイル内で使用されている単語を1次元リスト化
mahala_list = data.splitlines()
mahala_list = [sentence.replace(","," ").split() for sentence in mahala_list]

#著者毎の検証作品のマハラノビス距離を格納する変数
dist = {}

for cate in [cate1,cate2]:
  dist[cate] = []

#マハラノビス距離の格納
for line in mahala_list:
  for cate in [cate1,cate2]:
    if(int(line[2])==cate):
      val = float(line[1])*order
      dist[cate].append(val)

#ユークリッド距離のラベル
for cate in [cate1,cate2]:
  dist[cate] = np.array(list(map(float,dist[cate])))

print(len(dist[cate2]))

print("plot distance")
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
out_dir = "../result/euclid/nolta/"

use_color_list=[]
for author in authors:
  use_color_list.append(color_list[allauthors.index(author)])


cate_color = {913:(0.90,0.60,0),914:(0, 0.45, 0.7),911:(0, 0.6, 0.5)} # オレンジ、青、青みの強いみどり
cate_name = {913:"novel",914:"essay",911:"poetry"}
print(list( dist.values() ))

#　リスト,　bin=棒の数, rwidth=棒の幅, color=色, ラベル, histtype='bar' ,stacked=True(積み上げる)

### 対数表示用
ax.hist(list( dist.values() ), label=[cate_name[cate1],cate_name[cate2]],color=[cate_color[cate1],cate_color[cate2]],bins=np.logspace(0, 7, 20), range=(0, histmax), log=True ,rwidth=0.8,  histtype='bar', stacked=True)
ax.set_xscale("log")

### 普通
#ax.hist(list( dist.values() ), label=[cate_name[cate1],cate_name[cate2]],color=[cate_color[cate1],cate_color[cate2]],bins=20, range=(0, histmax),rwidth=0.8,  histtype='bar', stacked=True)


#ax.hist(main_distL, label=authors,color=color_list[allauthors.index(main_author)],bins=20, range=(0, histmax),rwidth=0.8,  histtype='bar', stacked=True,hatch='/',alpha=0.5)

ax.set_xlabel("Mahalanobis' distance",fontsize=15)
#ax.set_xlabel("Euclid distance",fontsize=15)
ax.set_ylabel('Number of works',fontsize=15)
ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left',borderaxespad=0, fontsize=15)
ax.tick_params(labelsize=15)
#fig.show()
print("saving glaph to png file")
fig.savefig(out_dir+output_file_name, bbox_inches="tight")


print("done")
