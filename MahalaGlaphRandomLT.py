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
what = "RandomMahalaGlaphLT" #mainauthorの学習作品120と，他の著者のテスト作品20*9
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
data = m.readFile(main_author,dic,"RandomMahala",target,vector,window,epoc,sep,model_target,other,extension,"../result/mahala/random/")

#ファイル内で使用されている単語を1次元リスト化
mahala_listT = data.splitlines()


print("plot mahala distance")
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
out_dir = "../result/mahala/random/"


#ファイル読み出し
dataL = m.readFile(main_author,dic,"RandomMahala",cate+"Learn",vector,window,epoc,sep,model_target,other,extension,"../result/mahala/random/")
#ファイル内で使用されている単語を1次元リスト化
mahala_listL = dataL.splitlines()

mahala_list = mahala_listL + mahala_listT

mahala_list = np.array(list(map(float,mahala_list)))

#　リスト,　bin=棒の数, rwidth=棒の幅, color=色, ラベル, histtype='bar' ,stacked=True(積み上げる)
ax.hist(mahala_list, bins=20, range=(0, histmax),rwidth=0.8,  histtype='bar', stacked=True)
ax.set_xlabel("Mahalanobis' Distance",fontsize=15)
ax.set_ylabel('Number of works',fontsize=15)
ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left',borderaxespad=0, fontsize=15)
ax.tick_params(labelsize=15)
#fig.show()
print("saving glaph to png file")
fig.savefig(out_dir+output_file_name, bbox_inches="tight")

print("done")