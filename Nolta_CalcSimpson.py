import Modules as m
import copy
import csv
from tqdm import tqdm
import re
import MeCab as mecab
import sys

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

#その他情報(rowなど)
sep = "#"
other = "alpha-worker1-symbol"
dirnum=""
cate1=913
cate2=914
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
all = {}
##==============================================

# メイン著者名とターゲットのカテゴリーを指定（小説じゃない方）
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
    print("not input cate2")
    exit()
model_target = "200work"+str(cate1)+"and"+str(cate2)

## 全語彙のSimpson係数とダイス係数の計算結果をいれる配列の初期化
resultS=[]
resultD=[]


## 品詞毎のSimpson係数とダイス係数の計算結果をいれる配列の初期化
HresultS = {}
HresultD = {}
for Hinshi in Hinshis:
    HresultS[Hinshi] = {}
    HresultD[Hinshi] = {}


h_list = ["ALL"]
h_list.extend(Hinshis)
print(h_list)
## 学習作品,検証作品，重複語彙の品詞別全語彙数を格納するdicの初期設定
N_vocab_num = {}
O_vocab_num = {}
NO_vocab_num = {}
for Hinshi in h_list:
    N_vocab_num[Hinshi] = None
    O_vocab_num[Hinshi] = None
    NO_vocab_num[Hinshi] = None

#　著者毎に処理する
N_file_names = []
O_file_names = []
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Make object for Main  "+main_author+"  $$$$$$$$$$$$$$$$$$$$$$$$$")
##  result/mahala/targetファイルから学習作品、検証作品のファイル名リストを生成
with open("../result/svm/nolta/target/"+m.nameFile(main_author,dic,"List",cate2,vector,window,epoc,"7030",model_target,other,".csv")) as f:
    reader = csv.reader(f)
    next = 0
    ##  学習作品を求める
    for row in reader:
        if(row[0]==str(cate1)+"_file_list"):
            next = 1
        elif(next==1):
            N_file_names = row
            next = 0       
        if(row[0]==str(cate2)+"_file_list"):
            next = 2
        elif(next==2):
            O_file_names = row
            next = 0
            break

print(len(N_file_names),len(O_file_names))


##　学習作品のファイル名から内容を読み出す
N_data = ""
#print(len(L_file_names))
for file_name in N_file_names:
    file = open(file_name)
    N_data += file.read()
print("    load "+str(len(N_file_names))+" file is done")
##  学習作品のファイル内で使用されている単語を1次元リスト化
N_word_list = N_data.replace("\n"," ").split()
if len( N_word_list ) == 0:
    print("[WARNING] File("+file_name+") 's word_list length is Zero")
# 単語の重複を許さず，語彙とする
N_word_list = list(set(N_word_list))

N_vocab_num["ALL"] = len(N_word_list)


##  各著者の検証作品についても同様に単語リストを生成する
#　検証作品のファイル名から内容を読み出す
O_data = ""
for file_name in O_file_names:
    file = open(file_name)
    O_data += file.read()
print("    load "+str(len(O_file_names))+" file is done")

#  検証作品のファイル内で使用されている単語を1次元リスト化
O_word_list = O_data.replace("\n"," ").split()
if len( O_word_list ) == 0:
    print("[WARNING] File("+file_name+") 's word_list length is Zero")

# 単語の重複を許さず，語彙とする
O_word_list = list(set(O_word_list))
O_vocab_num["ALL"] = len(O_word_list)

##　学習作品と検証作品の両方で使用されている語彙の集合を求める
N_O_wordlist_and = set(N_word_list) & set(O_word_list)
N_O_wordlist_and = list(N_O_wordlist_and)
NO_vocab_num["ALL"] = len(N_O_wordlist_and)

    # 学習作品の語彙数
    #print(len(N_word_list))
    # 検証作品の語彙数
    #print(len(O_word_list))
    # 共通の語彙数        
    #print(len(N_O_wordlist_and))

try:
    ##  Simpson係数の計算  
    # 学習作品と検証作品の語彙数の少ない方を分母とする
    resultS = float(float(len(N_O_wordlist_and))/float(min([len(N_word_list),len(O_word_list)])))
except:
    print(target,main_author)
    print("cannot claclation. denominator =",float(min([len(N_word_list),len(O_word_list)])))
    resultS = None

try:
    ##  Dice係数の計算  
    # 学習作品と検証作品の平均語彙数を分母とする
    resultD = float(float(2.0*len(N_O_wordlist_and))/float(len(N_word_list)+len(O_word_list)))
except:
    print(target,main_author)
    print("cannot claclation. denominator =",float(len(N_word_list)+len(O_word_list)))
    resultD = None


## MeCab で形態素解析した結果から，品詞，品詞細分類1,品詞細分類2,品詞細分類3を返す関数
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

##  品詞に対してループする
O_th_wordlist = {}
for target in Hinshis:
    ##  学習作品の単語リストから特定の品詞の単語を抜き出す
    N_th_wordlist = []
    for i in range(0,len(N_word_list)):
        hinshis = get_pos(N_word_list[i])
        if(hinshis[0] == target):
            N_th_wordlist.append(N_word_list[i])
    N_vocab_num[target] = len(N_th_wordlist)
    #print(len(N_th_wordlist))
    #print(target)

    
    # 検証作品のtarget品詞の単語リスト
    O_th_wordlist = []
    for i in range(0,len(O_word_list)):
        hinshis = get_pos(O_word_list[i])
        if(hinshis[0] == target):
            O_th_wordlist.append(O_word_list[i])
    #print(len(O_th_wordlist))
    O_vocab_num[target] = len(O_th_wordlist)

    ##　学習作品と検証作品の両方で使用されている語彙の集合を求める
    L_T_th_wordlist_and = set(N_th_wordlist) & set(O_th_wordlist)
    L_T_th_wordlist_and = list(L_T_th_wordlist_and)
    NO_vocab_num[target] = len(L_T_th_wordlist_and)
    
    try:
        ##  Simpson係数の計算  
        # 学習作品と検証作品の語彙数の少ない方を分母とする
        HresultS[target] = float(float(len(L_T_th_wordlist_and))/float( min([len(N_th_wordlist),len(O_th_wordlist)]) ) )
    except:
        print(target,main_author)
        print("cannot claclation. denominator =",float(min([len(N_th_wordlist),len(O_th_wordlist)])))
        HresultS[target] = None

    try:
        ##  Dice係数の計算  
        # 学習作品と検証作品の平均語彙数を分母とする
        HresultD[target] = float(float(2.0*len(L_T_th_wordlist_and))/float(len(N_th_wordlist)+len(O_th_wordlist)))
    except:
        print(target,main_author)
        print("cannot claclation. denominator =",float(len(N_th_wordlist)+len(O_th_wordlist)))
        HresultD[target] = None


##  Simpson係数の計算結果の出力
# 出力ファイルの指定
m.resetFile("../result/simpson/nolta/"+m.nameFile(main_author,dic,"Simpson",model_target,"#","#","#",sep,"#",other,".csv"))
outS = open("../result/simpson/nolta/"+m.nameFile(main_author,dic,"Simpson",model_target,"#","#","#",sep,"#",other,".csv"),"a")
# 全語彙（重複なし）での計算結果出力
print("All vocablalies",file=outS)
#タイトル行
m.printFixedLength("Simpson",12,endword=",",fileobj=outS)
m.printFixedLength("O_vocab_num",12,endword=",",fileobj=outS)
m.printFixedLength("N_vocab_num",12,endword=",",fileobj=outS)
m.printFixedLength("NO_vocab_num",16,endword=",",fileobj=outS)
print("",file=outS)    
#値
if(resultS != None):
    m.printFixedLength(round(resultS,4),12,endword=",",fileobj=outS)
else:
    m.printFixedLength(resultS,12,endword=",",fileobj=outS)
m.printFixedLength(O_vocab_num["ALL"],12,endword=",",fileobj=outS)
m.printFixedLength(N_vocab_num["ALL"],12,endword=":,",fileobj=outS)
m.printFixedLength(NO_vocab_num["ALL"],16,endword=",",fileobj=outS)
print("",file=outS)
print("",file=outS)


# 各品詞毎での計算結果出力
for Hinshi in Hinshis:
    # 品詞タイトル出力
    print(Hinshi+" vocablalies",file=outS)
    #タイトル行
    m.printFixedLength("Simpson",12,endword=",",fileobj=outS)
    m.printFixedLength("O_vocab_num",12,endword=",",fileobj=outS)
    m.printFixedLength("N_vocab_num",12,endword=",",fileobj=outS)
    m.printFixedLength("NO_vocab_num",16,endword=",",fileobj=outS)
    print("",file=outS)  
    #値
    if(HresultS[Hinshi] != None):
        m.printFixedLength(round(HresultS[Hinshi],4),10,endword=",",fileobj=outS)
    else:
        m.printFixedLength(HresultS[Hinshi],10,endword=",",fileobj=outS)
    m.printFixedLength(O_vocab_num[Hinshi],12,endword=",",fileobj=outS)
    m.printFixedLength(N_vocab_num[Hinshi],12,endword=":,",fileobj=outS)
    m.printFixedLength(NO_vocab_num[Hinshi],16,endword=",",fileobj=outS)
    print("",file=outS)
    print("",file=outS)

outS.close



##  Dice係数の計算結果の出力
# 出力ファイルの指定
m.resetFile("../result/dice/nolta/"+m.nameFile(main_author,dic,"Dice",model_target,"#","#","#",sep,"#",other,".csv"))
outD = open("../result/dice/nolta/"+m.nameFile(main_author,dic,"Dice",model_target,"#","#","#",sep,"#",other,".csv"),"a")
# 全語彙（重複なし）での計算結果出力
print("All vocablalies",file=outD)
#タイトル行
m.printFixedLength("Dice",12,endword=",",fileobj=outD)
m.printFixedLength("O_vocab_num",12,endword=",",fileobj=outD)
m.printFixedLength("N_vocab_num",12,endword=",",fileobj=outD)
m.printFixedLength("NO_vocab_num",16,endword=",",fileobj=outD)
print("",file=outD)    
#値
if(resultD != None):
    m.printFixedLength(round(resultD,4),12,endword=",",fileobj=outD)
else:
    m.printFixedLength(resultD,12,endword=",",fileobj=outD)
m.printFixedLength(O_vocab_num["ALL"],12,endword=",",fileobj=outD)
m.printFixedLength(N_vocab_num["ALL"],12,endword=":,",fileobj=outD)
m.printFixedLength(NO_vocab_num["ALL"],16,endword=",",fileobj=outD)
print("",file=outD)
print("",file=outD)

# 各品詞毎での計算結果出力
for Hinshi in Hinshis:
    print(Hinshi+" vocablalies",file=outD)
    #タイトル行
    m.printFixedLength("Dice",12,endword=",",fileobj=outD)
    m.printFixedLength("O_vocab_num",12,endword=",",fileobj=outD)
    m.printFixedLength("N_vocab_num",12,endword=",",fileobj=outD)
    m.printFixedLength("NO_vocab_num",16,endword=",",fileobj=outD)
    print("",file=outD)
    #値
    if(HresultD[Hinshi] != None):
        m.printFixedLength(round(HresultD[Hinshi],4),12,endword=",",fileobj=outD)
    else:
        m.printFixedLength(HresultD[Hinshi],12,endword=",",fileobj=outD)
    m.printFixedLength(O_vocab_num[Hinshi],12,endword=",",fileobj=outD)
    m.printFixedLength(N_vocab_num[Hinshi],12,endword=":,",fileobj=outD)
    m.printFixedLength(NO_vocab_num[Hinshi],16,endword=",",fileobj=outD)
    print("",file=outD)
    print("",file=outD)
