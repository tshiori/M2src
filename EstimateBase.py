import numpy as np
import matplotlib.pyplot as plt
import sys
import Modules as m
from classtest import Estimate

#著者名リスト
authors = ["Akutagawa","Arisima","Kajii","Kikuchi","Sakaguchi","Dazai","Nakajima","Natsume","Makino","Miyazawa"]
#辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']

##パラメータ-------------------------------------------
#ファイル情報
dic = "uNeologd"
main_authors = ["Akutagawa","Makino","Sakaguchi","Dazai","Miyazawa"]
what = "Mahala"
targets = ["ALL","Learn","Test"] 
target = targets[2] #Test
#モデルの学習パラメータ
vector = 100
window = 5
epoc =300
model_target = "ALLCorpus"
#コーパス分割割合，その他
sep = 100
other = "#"
#拡張子
extension =".csv"

#マハラノビス距離の分布よる推定で使用する閾値
#mahala_border = { "Akutagawa":240,"Makino":850,"Sakaguchi":320 }
mahala_border = { "Akutagawa":1480,"Makino":5390,"Sakaguchi":1820,"Dazai":6530,"Miyazawa":3980 }

#----------------------------------------------------------------

def numRest():
  num = {}
  for author in main_authors:
    num[author] = 0  
  return num

#初期化
authors_file_num = {}
for author in authors:
  authors_file_num[author] = 0

#著者毎の検証作品のマハラノビス距離を格納する変数
#mahala_list[メイン著者名][テスト著者名,メイン著者からのマハラノビス距離,ファイル名]
mahala_lists = {}
for author in main_authors:
  mahala_lists[author] = [0]
  data = m.readFile(author,dic,what,target,vector,window,epoc,sep,model_target,other,extension,"../result/mahala/")
  mahala_lists[author] = data.splitlines()
  mahala_lists[author] = [sentence.replace(","," ").split() for sentence in mahala_lists[author]]


#[著者名,芥川dist,牧野dist,坂口dist,ファイル名]の2重リスト構造作成
all_mahala_list = {}
file_num = len(mahala_lists["Akutagawa"])
for i in range(0,file_num):
  file_name = mahala_lists["Akutagawa"][i][2]
  all_mahala_list[i] = [mahala_lists["Akutagawa"][i][0]]
  for author in main_authors:
    if (file_name != mahala_lists[author][i][2]):
      print("[ERROR] "+author+"'s File name is wrong : index "+str(i))
      exit()
    all_mahala_list[i].append(mahala_lists[author][i][1])
  all_mahala_list[i].append(file_name)

file_obj={}
Estimate.main_authors = main_authors
for i in range(0,file_num):
    file_obj[i] = Estimate()
    file_obj[i].file_name=all_mahala_list[i][len(main_authors)+1]
    file_obj[i].correct_author=all_mahala_list[i][0]
    
    for main_author in main_authors:
      file_obj[i].mahala_dist[main_author]=all_mahala_list[i][int(main_authors.index(main_author))+1]

    #著者毎の作品数の集計
    for author in authors:
      if(file_obj[i].correct_author == author):
        authors_file_num[author] += 1


    file_obj[i].estimateResultMin()
    file_obj[i].estimateResultDistribution(mahala_border)
    file_obj[i].calcResultCode()
    file_obj[i].calcResultMatch()

print("著者のファイル数")
print(authors_file_num)



#マハラノビス距離の最小値による著者推定結果の集計===========================================================
#min_dist_num_book[テスト著者名][メイン著者1の数，メイン著者2の数，メイン著者3の数...]

min_dist_num_book={}
num=numRest()
#for test_author in authors:
for test_author in main_authors:
  min_dist_num_book[test_author]=[]
  for i in range(0,file_num):
    if(file_obj[i].correct_author==test_author):
      for main_author in main_authors:
        if(file_obj[i].result_min==main_author):
          num[main_author] += 1
  for main_author in main_authors:
    min_dist_num_book[test_author].append(num[main_author])
  num=numRest()

#推定結果ブック[テスト著者名][メイン著者1の数，メイン著者2の数，メイン著者3の数...]のコンソール出力
minEstF = open("../result/estimate/"+m.nameFile("MainAuthor",dic,"MinEstimate",target,vector,window,epoc,sep,model_target,other,".txt"),"w")
print("最小値推定結果",file=minEstF)
#for test_author in authors:
for test_author in main_authors:
  print(test_author+":"+str(min_dist_num_book[test_author]),test_author,authors_file_num[test_author],file=minEstF)
minEstF.close

#========================================================================================================

#マハラノビス距離の閾値による著者推定結果の集計===========================================================
#min_distri_num_book[テスト著者名][メイン著者1の数，メイン著者2の数，メイン著者3の数...]
#authors_file_num[著者名]=作品数　著者毎の全（ターゲット）作品数
book_distri_auth_num = {}
result_distri_main_num = {}
result_distri_other_num = 0
result_distri_main_correct_num = {}
distri_ffr = {}
distri_far = {}
for main_author in main_authors:
  #変数の初期化
  result_distri_main_num[main_author] = 0
  result_distri_main_correct_num[main_author] = 0

for i in range(0,file_num):
  #メイン著者であると判定された数　book_distri_auth_num[i]
  book_distri_auth_num[i] = 0
  for main_author in main_authors:
    if(file_obj[i].result_distri[main_author] == main_author):
      book_distri_auth_num[i] += 1

  #推定結果が1著者のみ
  if(book_distri_auth_num[i] == 1):
    #print("one")
    for main_author in main_authors:
      #メイン著者であると判定された作品数
      if(file_obj[i].result_distri[main_author] == main_author):
        #print("  "+main_author)
        file_obj[i].result_distri_val = main_author
        result_distri_main_num[main_author] += 1
        #メイン著者であると判定された作品の中でメイン著者の作品数
        if(file_obj[i].correct_author == main_author):
          result_distri_main_correct_num[main_author] += 1
  #推定結果がその他しかない
  elif(book_distri_auth_num[i] == 0):
    result_distri_other_num += 1
    file_obj[i].result_distri_val = "other"
  #推定結果が複数著者の場合
  else:
    result=9999.0#一時的な変数
    #最小のマハラノビス距離の著者を推定結果の著者とする
    for main_author in main_authors:
      if(file_obj[i].result_distri[main_author] == main_author and float(file_obj[i].mahala_dist[main_author])<float(result)):
        result = file_obj[i].mahala_dist[main_author]
        file_obj[i].result_distri_val = main_author
    #集計
    for main_author in main_authors:
      #メイン著者であると判定された作品数
      if(file_obj[i].result_distri_val == main_author):
        result_distri_main_num[main_author] += 1
        #メイン著者であると判定された作品の中でメイン著者の作品数
        if(file_obj[i].correct_author == main_author):
          result_distri_main_correct_num[main_author] += 1



print("閾値推定結果\nmain num")
print(result_distri_main_num)
print("main = correct num")
print(result_distri_main_correct_num)
print("others num")
print(result_distri_other_num)


#FFRとFARの算出
for main_author in main_authors:
  #本人拒否率FRR 
  distri_ffr[main_author] = float(float(authors_file_num[main_author] - result_distri_main_correct_num[main_author]) / float(authors_file_num[main_author]))
  #他人受入率FAR
  distri_far[main_author] = float(float(result_distri_main_num[main_author] - result_distri_main_correct_num[main_author]) / float(file_num - authors_file_num[main_author]))

  #正答率と推定精度
  #distri_p[main_author] = float(float(result_distri_main_correct_num[main_author]) / float(authors_file_num[main_author]))
  #distri_q[main_author] = float(float(result_distri_main_correct_num[main_author]) / float(result_distri_main_num[main_author]))

'''debug---
print("閾値推定結果確認\nmain num")
debug_main_num = {}
for main_author in main_authors:
  #変数の初期化
  debug_main_num[main_author] = 0
  for i in range(0,file_num):
    #メイン著者であると判定された作品数
    if(file_obj[i].result_distri[main_author] == main_author):
      debug_main_num[main_author] += 1
print(debug_main_num)
'''

'''---old---
#マハラノビス距離の閾値による著者推定結果の集計===========================================================
#min_distri_num_book[テスト著者名][メイン著者1の数，メイン著者2の数，メイン著者3の数...]
result_distri_main_num = {}
result_distri_main_correct_num = {}
authors_file_num = {}
distri_p = {}
distri_q = {}
for main_author in main_authors: 
  #変数の初期化
  result_distri_main_num[main_author] = 0
  result_distri_main_correct_num[main_author] = 0
  authors_file_num[main_author] = 0
  for i in range(0,file_num):
    #メイン著者であると判定された作品数
    if(file_obj[i].result_distri[main_author] == main_author):
      result_distri_main_num[main_author] += 1
    #メイン著者の全作品数
    if(file_obj[i].correct_author == main_author):
      authors_file_num[main_author] += 1
    #メイン著者であると判定された作品の中でメイン著者の作品数
    if(file_obj[i].result_distri[main_author] == main_author and file_obj[i].correct_author == main_author):
      result_distri_main_correct_num[main_author] += 1

  #本人拒否率FRR 
  distri_p[main_author] = float(float(authors_file_num[main_author] - result_distri_main_correct_num[main_author]) / float(authors_file_num[main_author]))
  #他人受入率FAR
  distri_q[main_author] = float(float(result_distri_main_num[main_author] - result_distri_main_correct_num[main_author]) / float(file_num - authors_file_num[main_author]))

  #正答率と推定精度
  #distri_p[main_author] = float(float(result_distri_main_correct_num[main_author]) / float(authors_file_num[main_author]))
  #distri_q[main_author] = float(float(result_distri_main_correct_num[main_author]) / float(result_distri_main_num[main_author]))
'''


### 上記２推定結果の集計============================================================================================
#result_code 表
#円グラフはエクセルとかでつくればいいかな、matplotでできるかな
result_code_num = {}
result_match_num = {}
result_match_correct_num = {}
for main_author in main_authors: 
  result_code_num[main_author] = [0,0,0,0]
  result_match_num[main_author] = 0
  result_match_correct_num[main_author] = 0
  for i in range(0,file_num):
    if(file_obj[i].result_match[main_author]==True):
      result_match_num[main_author] += 1
    if(file_obj[i].result_match[main_author]==True and file_obj[i].result_code[main_author]==3):
      result_match_correct_num[main_author] += 1
    for code in range(0,4):
      if(file_obj[i].result_code[main_author]==code and file_obj[i].correct_author == main_author):
        result_code_num[main_author][code] += 1
##====================================================================================================================


##最小値推定結果がメイン著者の作品の中で閾値より下の作品の集計========================================================
min_estimate_num = {}
min_estimate_u_border_num = {}
min_estimate_u_border_correct_num = {}
for main_author in main_authors: 
  min_estimate_num[main_author] = 0
  min_estimate_u_border_num[main_author] = 0
  min_estimate_u_border_correct_num[main_author] = 0
  for i in range(0,file_num):
    #最小値推定結果がメイン著者の作品数
    if(file_obj[i].result_min==main_author):
      min_estimate_num[main_author] += 1       

      #最小値推定結果がメイン著者の作品の中で閾値を下回る作品数
      if(file_obj[i].result_distri[main_author]==main_author):
        min_estimate_u_border_num[main_author] += 1

        #最小値推定結果がメイン著者の作品の中で閾値を下回る中で正解の作品数
        if(file_obj[i].correct_author==main_author):
            min_estimate_u_border_correct_num[main_author] += 1

##====================================================================================================================



# 円グラフを描画======================================================================================================
plt.rcParams['font.family'] = 'IPAexGothic'
plt.tick_params(labelsize=40)
plt.rcParams["font.size"] = 20
pie_label = ["両方の推定で不正解","最小値による推定では正解","閾値による推定では正解","両方の推定で正解"]
for main_author in main_authors:
  plt.figure(figsize=(20, 12), dpi=300)
  x = np.array(result_code_num[main_author])
  plt.pie(x, counterclock=False, startangle=90, autopct="%.1f%%",pctdistance=0.7)
  plt.legend(pie_label, fontsize=20,bbox_to_anchor=(0.9, 0.7)) 
  plt.savefig("../result/estimate/"+m.nameFile(main_author,dic,"resultCodePieGraph","Test",vector,window,epoc,sep,model_target,other,".png"), bbox_inches='tight')
##====================================================================================================================


#print(result_code_num)
#print(result_match_num)
#print(result_match_correct_num)

## 最小値推定のcsv出力================================================================================================
out_dir="../result/estimate/"
out_file=out_dir+m.nameFile(str(len(main_authors))+"author",dic,"Estimate",target,vector,window,epoc,sep,model_target,other,".csv")
m.resetFile(out_file)
m.csvWriting(out_file,["全ファイル数",file_num])

title=["最小値推定結果"]
m.csvWriting(out_file,title)
label = ["検証作品の著者名", "FRR[%]","FAR[%]"]
label[1:1] = main_authors
m.csvWriting(out_file,label)
line = []
for test_author in main_authors:
  min_dist_num_book[test_author][0:0] = [test_author]
  if(test_author in main_authors):
    print(test_author)
    print(sum(min_dist_num_book[test_author][1:]))
    print(min_dist_num_book[test_author][(main_authors.index(test_author))+1])
    frr = float(float(sum(min_dist_num_book[test_author][1:])-min_dist_num_book[test_author][main_authors.index(test_author)+1])/float(authors_file_num[test_author]))*100
    print(frr)
  else:
    frr = "-"
  min_dist_num_book[test_author].append(frr)
  m.csvWriting(out_file,min_dist_num_book[test_author])
##====================================================================================================================


##分布推定のcsv出力===================================================================================================
title=["\n\n閾値推定結果","border="+str(mahala_border)]
m.csvWriting(out_file,title)
label = ["メイン著者名","推定結果=メイン著者である作品数","推定結果=メイン著者の作品数","メイン著者の検証作品数","FFR[%]","FAR[%]"]
m.csvWriting(out_file,label)
for main_author in main_authors:
  row = [main_author,result_distri_main_correct_num[main_author],result_distri_main_num[main_author],authors_file_num[main_author],distri_ffr[main_author]*100,distri_far[main_author]*100]
  m.csvWriting(out_file,row)
m.csvWriting(out_file,["\n推定結果がどの著者でもない作品数：",result_distri_other_num])
##=====================================================================================================================

##最小値推定結果がメイン著者の作品の中で閾値より下の作品===============================================================
title=["\n\n最小値推定結果がメイン著者の作品の中で閾値より下の作品","border="+str(mahala_border)]
m.csvWriting(out_file,title)
label = ["メイン著者名","最小値推定結果=メイン著者である作品数","最小値推定結果=メイン著者であり閾値を下回っている作品数","最小値推定結果=メイン著者であり閾値を下回っている作品の中で正しい推定結果の作品数","FRR[%]","FAR[%]"]
m.csvWriting(out_file,label)
p=""
q=""
for main_author in main_authors:
  p = float(float(authors_file_num[main_author] - min_estimate_u_border_correct_num[main_author]) / float(authors_file_num[main_author]))*100
  if(min_estimate_u_border_num[main_author]!=0):
    q = float(float(min_estimate_u_border_num[main_author] - min_estimate_u_border_correct_num[main_author]) / float(file_num - authors_file_num[main_author]))*100
  row = [main_author,min_estimate_num[main_author],min_estimate_u_border_num[main_author],min_estimate_u_border_correct_num[main_author],p,q]
  m.csvWriting(out_file,row)
##=====================================================================================================================


##推定結果の一致と正答率のcsv出力=====================================================================================
title=["\n\n推定結果の一致と正答率の関係","border="+str(mahala_border)]
m.csvWriting(out_file,title)
label = ["メイン著者名","推定結果が一致する作品数","両方の推定結果が正しい作品数"]
m.csvWriting(out_file,label)
for main_author in main_authors:
  row = [main_author,result_match_num[main_author],result_match_correct_num[main_author]]
  m.csvWriting(out_file,row)
##=====================================================================================================================