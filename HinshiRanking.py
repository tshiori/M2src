import Modules as m
import copy
import csv
from tqdm import tqdm
import re

###定義と初期化---------------------------------------------------------------
##辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']

##パラメータ設定==============================
#辞書名
dic = dics[3] #uNEologd
#何のファイルか
what = "CountHinshi"
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
sep = "#"
other = "alpha-worker1"
vocab = True
dirnum=""
cate=913
#拡張子
extension = ".csv"
##==============================================

##その他初期化
#著者名リスト
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
main_authors = ['Akutagawa', 'Sakaguchi', 'Makino']
##　品詞リスト
Hinshis = ["名詞","動詞","形容詞","副詞","連体詞","助詞","助動詞","接続詞","記号","感動詞","接頭詞","フィラー","その他"]
Hinshis_plus = copy.copy(Hinshis)
##  品詞ランキング格納辞書の初期化
Hinshi_ranking_dic = {}
Detail1_ranking_dic = {}
Detail2_ranking_dic = {}
Detail3_ranking_dic = {}
all = {}
##==============================================

## 学習作品の判定のときは作品数の多いメイン著者だけを対象著者とする
if (target == targets[1]):
    authors = main_authors
target = str(cate) + target

#　著者毎に処理する
for author in authors:
    print("=========================="+author+"===============================")
    ##  対象の決定   
    if(target == str(cate) + targets[0]):
        ##　対象著者のデータベースから全作品のインスタンス生成してファイル名リストを取得
        #インスタンス生成のための各変数の初期化
        all_works = {}
        #各著者のインスタンスを生成する
        all_works,index_fin = m.getWorkObj(author,0)
        #print(len(all_works))
        #重複なし小説作品のインデックスを著者毎にリスト化する
        file_names=[]
        for index in range(0,len(all_works)):
            try:
                if(all_works[index].author != author):
                    print(all_works[index].author)
                if(all_works[index].duplicate==False and all_works[index].category==cate):
                    file_names.append(all_works[index].filepath)
            except:
                print(index)
                exit()
                break
        #print(len(file_names))
    elif(target == str(cate) + targets[1]):
        ##  result/mahala/targetファイルから学習作品のファイル名リストを生成
        with open("../result/mahala/target/"+m.nameFile(author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv")) as f:
            reader = csv.reader(f)
            next=0
            for row in reader:
                if(row[0]=='main_lerans_index_list'):
                    next=1
                elif(next==1):
                    file_names = row
                    break
    else:
        print(target)
        exit()
    #print(len(file_names))


    ##　対象ファイルのファイル名から内容を読み出す
    data = ""
    print(len(file_names))
    for file_name in tqdm(file_names):
        file = open(file_name)
        data += file.read()
    print("    load "+str(len(file_names))+" file is done")

    ## ファイル内で使用されている単語を1次元リスト化
    word_list = data.replace("\n"," ").split()
    if len( word_list ) == 0:
        print("[WARNING] File("+file_name+") 's word_list length is Zero")

    ## vocabフラグがTrueだったらword_listの重複を許さない＝語彙に対するカウント
    if( vocab == True ):
        word_list = list(set(word_list))

    ## MeCab で形態素解析した結果から，品詞，品詞細分類1,品詞細分類2,品詞細分類3を返す関数
    import MeCab as mecab
    tagger = mecab.Tagger("-p")
    ## 参考：https://analytics-note.xyz/programming/mecab-determine-pos/
    def get_pos(word):
        # 制約付き解析の形態素断片形式にする
        p_token = f"{word}\t*"
        ## print(p_token)
        ## result = 保吉	*
        parsed_line = tagger.parse(p_token).splitlines()[0]
        ## print(tagger.parse(p_token).splitlines()[0])
        ## result = 保吉	名詞,固有名詞,人名,名,*,*,保吉,ヤスキチ,ヤスキチ
        ##          表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
        feature = parsed_line.split("\t")[1]
        ## print(features)
        ## result = 名詞,固有名詞,人名,名,*,*,保吉,ヤスキチ,ヤスキチ
        
        # ,(カンマ)で区切り、品詞,品詞細分類1,品詞細分類2,品詞細分類3 の4項目残す
        pos_list = feature.split(",")[:4]
        #print(pos_list)

        return pos_list


    #品詞カウント辞書の定義
    Hinshi_count_dic = {}
    for Hinshi in Hinshis:
        Hinshi_count_dic[Hinshi] = 0
    #品詞細分類1のリストとカウント辞書の定義
    Detail1s = []
    Detail1_count_dic = {}
    #品詞細分類2のリストとカウント辞書の定義
    Detail2s = []
    Detail2_count_dic = {}
    #品詞細分類3のリストとカウント辞書の定義
    Detail3s = []
    Detail3_count_dic = {}


    ## 単語の品詞を判定（品詞，品詞細分類1，品詞細分類2，品詞細分類3）
    print("Determine the part of speechies")
    for i in tqdm(range(0,len(word_list))):
        hinshis = get_pos(word_list[i])
        # 品詞のカウント
        if (hinshis[0] in  Hinshis_plus):
            Hinshi_count_dic[hinshis[0]] += 1
        else:
            Hinshis_plus.append(hinshis[0])
            Hinshi_count_dic[hinshis[0]] = 0
        # 品詞-品詞細分類1のカウント
        if (hinshis[0]+"-"+hinshis[1] in Detail1s ):
            Detail1_count_dic[hinshis[0]+"-"+hinshis[1]] += 1
        else:
            Detail1s.append(hinshis[0]+"-"+hinshis[1])
            Detail1_count_dic[hinshis[0]+"-"+hinshis[1]] = 1
        # 品詞-品詞細分類1-品詞細分類2のカウント
        if (hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2] in Detail2s ):
            Detail2_count_dic[hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2]] += 1
        else:
            Detail2s.append(hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2])
            Detail2_count_dic[hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2]] = 1
        # 品詞-品詞細分類1-品詞細分類2-品詞細分類3のカウント
        if (hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2]+"-"+hinshis[3] in Detail3s ):
            Detail3_count_dic[hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2]+"-"+hinshis[3]] += 1
        else:
            Detail3s.append(hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2]+"-"+hinshis[3])
            Detail3_count_dic[hinshis[0]+"-"+hinshis[1]+"-"+hinshis[2]+"-"+hinshis[3]] = 1
        
    ## 品詞リストが更新されているかの確認
    if(Hinshis != Hinshis_plus):
        print("Updated Hinishi list")
        for Hinshi in Hinshis_plus:
            if Hinshi not in Hinshis:
                print(Hinshi)

    ## Count結果をcsvに出力
    #書き込みファイルの指定
    m.resetFile("../result/count/"+m.nameFile(author,dic,"HinshiCount",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"))
    outf = open("../result/count/"+m.nameFile(author,dic,"HinshiCount",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"),"a")
    print("Target = "+target+str(len(file_names))+"works",file=outf)
    if vocab==True:
        print("Word duplicate is disallowed. i.e. We count vocabulary.",file=outf)
    else:
        print("Word duplicate is allowed. i.e. We count number of words.",file=outf)
    print("all",end=",",file=outf)
    print(len(word_list),file=outf)
    all[author] = float(len(word_list))
    writer = csv.writer(outf)
    for Hinshi in tqdm(Hinshis_plus):
        #　品詞数
        row = [ Hinshi,Hinshi_count_dic[Hinshi], "" , "" , "" , "" , "" , "" ]
        writer.writerow(row)
        #  品詞-品詞細分類1数
        for Dtail1 in Detail1s:
            if(Dtail1.startswith(Hinshi)):
                row = [  "" , "" , Dtail1 , Detail1_count_dic[Dtail1] , "" , "" , "" , "" ]
                writer.writerow(row)
                #  品詞-品詞細分類1-品詞細分類2数
                for Dtail2 in Detail2s:
                    if(Dtail2.startswith(Dtail1)):
                        row = [  "" , "" , "" , "" , Dtail2 , Detail2_count_dic[Dtail2] , "" , "" ]
                        writer.writerow(row)
                        #  品詞-品詞細分類1-品詞細分類2-詞細分類3数
                        for Dtail3 in Detail3s:
                            if(Dtail3.startswith(Dtail2)):
                                row = [  "" , "" , "" , "" , "" , "" , Dtail3 , Detail3_count_dic[Dtail3] ]
                                writer.writerow(row)
    outf.close

    ## カウント結果を値に対して降順ソートし，使用回数順のランキング形式とする
    # 品詞ランキング
    Hinshi_ranking_dic[author] = sorted(Hinshi_count_dic.items(),key=lambda x:x[1],reverse=True)
    #print(Hinshi_ranking_dic[author])
    # 品詞細分類1ランキング
    Detail1_ranking_dic[author] = sorted(Detail1_count_dic.items(),key=lambda x:x[1],reverse=True)
    #print(Detail1_ranking_dic[author])
    # 品詞細分類2ランキング
    Detail2_ranking_dic[author] = sorted(Detail2_count_dic.items(),key=lambda x:x[1],reverse=True)
    #print(Detail2_ranking_dic[author])
    # 品詞細分類3ランキング
    Detail3_ranking_dic[author] = sorted(Detail3_count_dic.items(),key=lambda x:x[1],reverse=True)
    #print(Detail3_ranking_dic[author])
    


## 品詞ランキング結果をcsvに出力
# 書き込みファイルの指定
m.resetFile("../result/count/"+m.nameFile(len(authors),dic,"HinshiRanking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"))
Hrankf = open("../result/count/"+m.nameFile(len(authors),dic,"HinshiRanking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"),"a")
print("Target = "+target+str(len(file_names))+"works",file=Hrankf)
if vocab==True:
    print("Word duplicate is disallowed. i.e. We count vocabulary.",file=Hrankf)
else:
    print("Word duplicate is allowed. i.e. We count number of words.",file=Hrankf)
print("All Hinshi number is "+str(len(Hinshis_plus)),file=Hrankf)
# 著者の名前の行
print(",",end="",file=Hrankf)
for author in authors:
    print(author,end=", , ,", file = Hrankf)
print("",file=Hrankf)
# 順位
for i in list(range(1,len(Hinshis_plus)+1)):
    print(i,end=",",file=Hrankf)
    for author in authors:
        print(Hinshi_ranking_dic[author][i-1],end=",",file=Hrankf)
        print( str(round(float(float(Hinshi_ranking_dic[author][i-1][1])/all[author]*100),2))+str("%"),end=",",file=Hrankf )
    print("",file=Hrankf)
Hrankf.close


## 品詞細分類1ランキング結果をcsvに出力
# 品詞細分類1のが何種類あるか
length = []
for author in authors:
    length.append(len(Detail1_ranking_dic[author]))
length = max(length)
print("length=",length)
# 書き込みファイルの指定
m.resetFile("../result/count/"+m.nameFile(len(authors),dic,"HinshiDetail1Ranking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"))
Hrankf = open("../result/count/"+m.nameFile(len(authors),dic,"HinshiDetail1Ranking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"),"a")
print("Target = "+target+str(len(file_names))+"works",file=Hrankf)
if vocab==True:
    print("Word duplicate is disallowed. i.e. We count vocabulary.",file=Hrankf)
else:
    print("Word duplicate is allowed. i.e. We count number of words.",file=Hrankf)
print("All Hinshi Detail1 number is "+str(len(Hinshis_plus)),file=Hrankf)
# 著者の名前の行
print(",",end="",file=Hrankf)
for author in authors:
    print(author,end=", , ,", file = Hrankf)
print("",file=Hrankf)
# 順位
for i in list(range(1,length+1)):
    print(i,end=",",file=Hrankf)
    for author in authors:
        try:
            print(Detail1_ranking_dic[author][i-1],end=",",file=Hrankf)
            print( str(round(float(float(Detail1_ranking_dic[author][i-1][1])/all[author]*100),2))+str("%"),end=",",file=Hrankf )
        except:
            pass
    print("",file=Hrankf)
Hrankf.close


## 品詞細分類2ランキング結果をcsvに出力
# 品詞細分類1のが何種類あるか
length = []
for author in authors:
    length.append(len(Detail2_ranking_dic[author]))
length = max(length)
print("length=",length)
# 書き込みファイルの指定
m.resetFile("../result/count/"+m.nameFile(len(authors),dic,"HinshiDetail2Ranking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"))
Hrankf = open("../result/count/"+m.nameFile(len(authors),dic,"HinshiDetail2Ranking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"),"a")
print("Target = "+target+str(len(file_names))+"works",file=Hrankf)
if vocab==True:
    print("Word duplicate is disallowed. i.e. We count vocabulary.",file=Hrankf)
else:
    print("Word duplicate is allowed. i.e. We count number of words.",file=Hrankf)
print("All Hinshi number is "+str(len(Hinshis_plus)),file=Hrankf)
# 著者の名前の行
print(",",end="",file=Hrankf)
for author in authors:
    print(author,end=", , ,", file = Hrankf)
print("",file=Hrankf)
# 順位
for i in list(range(1,length+1)):
    print(i,end=",",file=Hrankf)
    for author in authors:
        try:
            print(Detail2_ranking_dic[author][i-1],end=",",file=Hrankf)
            print( str(round(float(float(Detail2_ranking_dic[author][i-1][1])/all[author]*100),2))+str("%"),end=",",file=Hrankf )
        except:
            pass
    print("",file=Hrankf)
Hrankf.close


## 品詞細分類3ランキング結果をcsvに出力
# 品詞細分類1のが何種類あるか
length = []
for author in authors:
    length.append(len(Detail3_ranking_dic[author]))
length = max(length)
print("length=",length)
# 書き込みファイルの指定
m.resetFile("../result/count/"+m.nameFile(len(authors),dic,"HinshiDetail3Ranking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"))
Hrankf = open("../result/count/"+m.nameFile(len(authors),dic,"HinshiDetail3Ranking",target,"#","#","#",sep,"#","vocab-"+str(vocab),".csv"),"a")
print("Target = "+target+str(len(file_names))+"works",file=Hrankf)
if vocab==True:
    print("Word duplicate is disallowed. i.e. We count vocabulary.",file=Hrankf)
else:
    print("Word duplicate is allowed. i.e. We count number of words.",file=Hrankf)
print("All Hinshi number is "+str(len(Hinshis_plus)),file=Hrankf)
# 著者の名前の行
print(",",end="",file=Hrankf)
for author in authors:
    print(author,end=", , ,", file = Hrankf)
print("",file=Hrankf)
# 順位
for i in list(range(1,length+1)):
    print(i,end=",",file=Hrankf)
    for author in authors:
        try:
            print(Detail3_ranking_dic[author][i-1],end=",",file=Hrankf)
            print( str(round(float(float(Detail3_ranking_dic[author][i-1][1])/all[author]*100),2))+str("%"),end=",",file=Hrankf )
        except:
            pass
    print("",file=Hrankf)
Hrankf.close


'''
print(Hinshi_count_dic)
print(Detail1_count_dic)
print(Detail2_count_dic)
print(Detail3_count_dic)
'''
