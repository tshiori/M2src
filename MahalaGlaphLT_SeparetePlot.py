'''
メイン著者の120作品の学習作品と、
その他の著者の20作品の検証作品のマハラノビス距離を同一グラフに描画
※メイン著者の検証作品20作品は描画しない
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
#authors = ["Akutagawa"]
auth_num = str(len(authors))
#グラフの色　　0        1           2         3       4       5           6          7        8      9
color_list = ['red','hotpink','darkorange','gold','skyblue','green','yellowgreen','brown','blue','black']#Testのとき
#color_list = ['red','skyblue','green','blue','black'] #Lernのとき
#辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']

##パラメータ-------------------------------------------
#ファイル情報
dic = "uNeologd"
main_author = ""
what = "MahalaGlaphLT" #mainauthorの学習作品120と，他の著者のテスト作品20*9
targets = ["ALL","Learn","Test","ALLCorpus","NoneNovel","Novel"]  
target = ""
#モデルの学習パラメータ
vector = 100
window = 5
epoc = 500
#使用モデルのが学習したファイル
model_targets = ["ALL","Learn","ALLCorpus","#"]
model_target = model_targets[2] #ALLCorpus
#コーパス分割割合，その他
#sep = 100
sep = "T20L120"
other = "alpha-worker1"
#拡張子
extension =".csv"

#グラフの右端マハラノビス距離
histmax = 0
#-----------------------------------------------------

### 入力例　mainauthor="Akutagawa" cate="913" histmax="10000"
main_author = input("input main author name =  ")
#main_author = "Akutagawa"
#cate = input("input category (NDC) =  ")
cate = "913"
target = str(cate)+"Test"
histmax = int(input("input histmax =  "))
output_file_name = m.nameFile(main_author,dic,what,target,vector,window,epoc,sep,model_target,"histmax"+str(histmax)+"-"+auth_num+"author",".png")
print(output_file_name)
m.continueOrExit()

#ファイル読み出し
data = m.readFile(main_author,dic,"Mahala",target,vector,window,epoc,sep,model_target,other,extension,"../result/mahala/")

#ファイル内で使用されている単語を1次元リスト化
mahala_list = data.splitlines()
mahala_list = [sentence.replace(","," ").split() for sentence in mahala_list]

#著者毎の検証作品のマハラノビス距離を格納する変数
auth_dist = {}
for author in authors:
    auth_dist[author] = []

#マハラノビス距離の格納
for line in mahala_list:
  for author in authors:
    if(line[0]==author):
      auth_dist[author].append(line[1])

#マハラノビス距離のリストの数値化
for author in authors:
  auth_dist[author] = np.array(list(map(float,auth_dist[author])))

print("plot mahala distance")
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
out_dir = "../result/mahala/"

use_color_list=[]
for author in authors:
  use_color_list.append(color_list[authors.index(author)])


#ファイル読み出し
dataL = m.readFile(main_author,dic,"Mahala",cate+"Learn",vector,window,epoc,sep,model_target,other,extension,"../result/mahala/")
#ファイル内で使用されている単語を1次元リスト化
mahala_listL = dataL.splitlines()
mahala_listL = [sentence.replace(","," ").split() for sentence in mahala_listL]
#マハラノビス距離の格納
main_distL = []
for line in mahala_listL:
  if(line[0]==main_author):
    main_distL.append(line[1])
  else:
    print("main author's learn works mahala dist is not found in...")
    print(m.nameFile(main_author,dic,"Mahala",cate+"Learn",vector,window,epoc,sep,model_target,other,extension))
    exit()
auth_dist[main_author] = np.append(auth_dist[main_author],np.array(list(map(float,main_distL))))

'''
import array
print(list( auth_dist.values() ))
x = list( auth_dist.values() )
x.extend([main_distL])
print(x)
exit()
'''

#バーが重複しない 2つのヒストグラム
#https://www.delftstack.com/ja/howto/matplotlib/how-to-plot-two-histograms-in-one-plot-in-matplotlib/



#　リスト,　bin=棒の数, rwidth=棒の幅, color=色, ラベル, histtype='bar' ,stacked=True(積み上げる)
ax.hist(list( auth_dist.values() ), label=authors,color=use_color_list,bins=20, range=(0, histmax),rwidth=0.8,  histtype='bar', stacked=True)
#ax.hist(main_distL, label=authors,color=color_list[authors.index(main_author)],bins=20, range=(0, histmax),rwidth=0.8,  histtype='bar', stacked=True,hatch='/',alpha=0.5)

ax.set_xlabel("Mahalanobis' Distance",fontsize=15)
ax.set_ylabel('Number of works',fontsize=15)
ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left',borderaxespad=0, fontsize=15)
ax.tick_params(labelsize=15)
#fig.show()

print("saving glaph to png file")
fig.savefig(out_dir+"TESTTTTTTTTTTTTTtt"+output_file_name, bbox_inches="tight")


print("done")
