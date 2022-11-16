import Modules as m
import copy
import csv
from tqdm import tqdm
import re
import MeCab as mecab

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
all = {}
##==============================================

## 対象著者は全著者とする。対象カテゴリを指定できる。
target = str(cate)



## 全語彙のSimpson係数とダイス係数の計算結果をいれる配列の初期化
resultS={}
resultD={}
for test_author in authors:
    resultS[test_author]={}
    resultD[test_author]={}


## 品詞毎のSimpson係数とダイス係数の計算結果をいれる配列の初期化
HresultS = {}
HresultD = {}
for Hinshi in Hinshis:
    HresultS[Hinshi] = {}
    HresultD[Hinshi] = {}
    for test_author in authors:
        HresultS[Hinshi][test_author] = {}
        HresultD[Hinshi][test_author] = {}


h_list = ["ALL"]
h_list.extend(Hinshis)
print(h_list)
## 学習作品,検証作品，重複語彙の品詞別全語彙数を格納するdicの初期設定
l_vocab_num = {}
t_vocab_num = {}
lt_and_vocab_num = {}
for Hinshi in h_list:
    l_vocab_num[Hinshi] = {}
    t_vocab_num[Hinshi] = {}
    lt_and_vocab_num[Hinshi] = {}
    for main_author in main_authors:
        l_vocab_num[Hinshi][main_author] = None
        t_vocab_num[Hinshi][main_author] = {}
        lt_and_vocab_num[Hinshi][main_author] = {}
        for test_author in authors:
            t_vocab_num[Hinshi][main_author][test_author] = None
            lt_and_vocab_num[Hinshi][main_author][test_author] = None
            


#　著者毎に処理する
T_file_names = {}
for main_author in main_authors:
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Make object for Main  "+main_author+"  $$$$$$$$$$$$$$$$$$$$$$$$$")
    ##  result/mahala/targetファイルから学習作品、検証作品のファイル名リストを生成
    with open("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv")) as f:
        reader = csv.reader(f)
        next = 0
        ##  学習作品を求める
        for row in reader:
            if(row[0]=='main_lerans_index_list'):
                next = 1
            elif(next==1):
                L_file_names = row
                next = 0
                break
        #print("L_length=",len(L_file_names))
        #aprint("L_file_names=",L_file_names[119])

        ##  検証作品を求める       
        for test_author in authors:
            for row in reader:
                if(row[0]=='test_index_list['+test_author+']'):
                    next = 1
                elif(next==1):
                    T_file_names[test_author] = row
                    next = 0
                    break
            #print(test_author,"length=",len(T_file_names[test_author]))
            #print("T_file_names=",T_file_names[test_author][0])


    ##　学習作品のファイル名から内容を読み出す
    L_data = ""
    #print(len(L_file_names))
    for file_name in L_file_names:
        file = open(file_name)
        L_data += file.read()
    print("    load "+str(len(L_file_names))+" file is done")
    ##  学習作品のファイル内で使用されている単語を1次元リスト化
    L_word_list = L_data.replace("\n"," ").split()
    if len( L_word_list ) == 0:
        print("[WARNING] File("+file_name+") 's word_list length is Zero")
    # 単語の重複を許さず，語彙とする
    L_word_list = list(set(L_word_list))
    l_vocab_num["ALL"][main_author] = len(L_word_list)


    ##  各著者の検証作品についても同様に単語リストを生成する
    T_word_list={}
    for test_author in authors:
        print("============ ALL Vocab ==========  "+test_author+"  =========================")
        #　検証作品のファイル名から内容を読み出す
        T_data = ""
        for file_name in T_file_names[test_author]:
            file = open(file_name)
            T_data += file.read()
        print("    load "+str(len(T_file_names[test_author]))+" file is done")

        #  検証作品のファイル内で使用されている単語を1次元リスト化
        T_word_list[test_author] = T_data.replace("\n"," ").split()
        if len( T_word_list[test_author] ) == 0:
            print("[WARNING] File("+file_name+") 's word_list length is Zero")

        # 単語の重複を許さず，語彙とする
        T_word_list[test_author] = list(set(T_word_list[test_author]))
        t_vocab_num["ALL"][main_author][test_author] = len(T_word_list[test_author])
        
        ##　学習作品と検証作品の両方で使用されている語彙の集合を求める
        L_T_wordlist_and = set(L_word_list) & set(T_word_list[test_author])
        L_T_wordlist_and = list(L_T_wordlist_and)
        lt_and_vocab_num["ALL"][main_author][test_author] = len(L_T_wordlist_and)

        # 学習作品の語彙数
        #print(len(L_word_list))
        # 検証作品の語彙数
        #print(len(T_word_list[test_author]))
        # 共通の語彙数        
        #print(len(L_T_wordlist_and))


        try:
            ##  Simpson係数の計算  
            # 学習作品と検証作品の語彙数の少ない方を分母とする
            resultS[test_author][main_author] = float(float(len(L_T_wordlist_and))/float(min([len(L_word_list),len(T_word_list[test_author])])))
        except:
            print(target,test_author,main_author)
            print("cannot claclation. denominator =",float(min([len(L_th_wordlist),len(T_th_wordlist[test_author])])))
            resultS[test_author][main_author] = None

        try:
            ##  Dice係数の計算  
            # 学習作品と検証作品の平均語彙数を分母とする
            resultD[test_author][main_author] = float(float(2.0*len(L_T_wordlist_and))/float(len(L_word_list)+len(T_word_list[test_author])))
        except:
            print(target,test_author,main_author)
            print("cannot claclation. denominator =",float(min([len(L_th_wordlist),len(T_th_wordlist[test_author])])))
            resultD[test_author][main_author] = None

    
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
    T_th_wordlist = {}
    for target in Hinshis:
        ##  学習作品の単語リストから特定の品詞の単語を抜き出す
        L_th_wordlist = []
        for i in range(0,len(L_word_list)):
            hinshis = get_pos(L_word_list[i])
            if(hinshis[0] == target):
                L_th_wordlist.append(L_word_list[i])
        l_vocab_num[target][main_author] = len(L_th_wordlist)
        #print(len(L_th_wordlist))
        #print(target)

       ##  各著者の検証作品の単語リストから特定の品詞の単語を抜き出す
        for test_author in authors:
            # 検証作品のtarget品詞の単語リスト
            T_th_wordlist[test_author] = []
            for i in range(0,len(T_word_list[test_author])):
                hinshis = get_pos(T_word_list[test_author][i])
                if(hinshis[0] == target):
                    T_th_wordlist[test_author].append(T_word_list[test_author][i])
            #print(len(T_th_wordlist[test_author]))
            t_vocab_num[target][main_author][test_author] = len(T_th_wordlist[test_author])

            ##　学習作品と検証作品の両方で使用されている語彙の集合を求める
            L_T_th_wordlist_and = set(L_th_wordlist) & set(T_th_wordlist[test_author])
            L_T_th_wordlist_and = list(L_T_th_wordlist_and)
            lt_and_vocab_num[target][main_author][test_author] = len(L_T_th_wordlist_and)
            

            try:
                ##  Simpson係数の計算  
                # 学習作品と検証作品の語彙数の少ない方を分母とする
                HresultS[target][test_author][main_author] = float(float(len(L_T_th_wordlist_and))/float( min([len(L_th_wordlist),len(T_th_wordlist[test_author])]) ) )
            except:
                print(target,test_author,main_author)
                print("cannot claclation. denominator =",float(min([len(L_th_wordlist),len(T_th_wordlist[test_author])])))
                HresultS[target][test_author][main_author] = None

            try:
                ##  Dice係数の計算  
                # 学習作品と検証作品の平均語彙数を分母とする
                HresultD[target][test_author][main_author] = float(float(2.0*len(L_T_th_wordlist_and))/float(len(L_th_wordlist)+len(T_th_wordlist[test_author])))
            except:
                print(target,test_author,main_author)
                print("cannot claclation. denominator =",float(len(L_th_wordlist)+len(T_th_wordlist[test_author])))
                HresultD[target][test_author][main_author] = None


##  Simpson係数の計算結果の出力
# 出力ファイルの指定
m.resetFile("../result/simpson/"+m.nameFile(len(authors),dic,"Simpson",cate,"#","#","#",sep,"#","#",".csv"))
outS = open("../result/simpson/"+m.nameFile(len(authors),dic,"Simpson",cate,"#","#","#",sep,"#","#",".csv"),"a")
print("../result/mahala/target/"+m.nameFile(main_author,dic,"List",cate,vector,window,epoc,"T20L120",model_target,other,".csv"),file=outS)
# 全語彙（重複なし）での計算結果出力
print("All vocablalies",file=outS)
print("          ",end=",",file=outS)
# 1行目列タイトル出力
for author in main_authors:
    m.printFixedLength(author,10,endword=",",fileobj=outS)
m.printFixedLength("T_vocab_num",12,endword=",",fileobj=outS)
m.printFixedLength("LT_and_vocab_num",16,endword=",",fileobj=outS)
print("",file=outS)
# 2行目以降検証著者名B，著者Bの検証作品と各メイン著者の学習作品語彙でのSimpson係数，検証語彙数，各重複語彙数
for test_author in authors:
    m.printFixedLength(test_author,10,endword=":,",fileobj=outS)
    for main_author in main_authors:
        if(resultS[test_author][main_author] != None):
            m.printFixedLength(round(resultS[test_author][main_author],4),10,endword=",",fileobj=outS)
        else:
            m.printFixedLength(resultS[test_author][main_author],10,endword=",",fileobj=outS)
    m.printFixedLength(t_vocab_num["ALL"][main_author][test_author],12,endword=",",fileobj=outS)
    for main_author in main_authors:
            m.printFixedLength(lt_and_vocab_num["ALL"][main_author][test_author],6,endword=",",fileobj=outS)
    print("",file=outS)
m.printFixedLength("L_vocab_num",11,endword=":,",fileobj=outS)
# 最終行各メイン著者の学習作品の語彙数
for main_author in main_authors:
    m.printFixedLength(l_vocab_num["ALL"][main_author],11,endword=",",fileobj=outS)
print("",file=outS)
print("",file=outS)

# 各品詞毎での計算結果出力
for Hinshi in Hinshis:
    # 品詞タイトル出力
    print(Hinshi+" vocablalies",file=outS)
    # 2行目列タイトル出力
    print("          ",end=",",file=outS)
    for author in main_authors:
        m.printFixedLength(author,10,endword=",",fileobj=outS)
    m.printFixedLength("T_vocab_num",12,endword=",",fileobj=outS)
    m.printFixedLength("LT_and_vocab_num",16,endword=",",fileobj=outS)
    print("",file=outS)
    # 3行目以降[ 検証著者名B，著者Bの検証作品と各メイン著者の学習作品語彙でのSimpson係数，検証語彙数，各重複語彙数]
    for test_author in authors:
        m.printFixedLength(test_author,10,endword=":,",fileobj=outS)
        for main_author in main_authors:
           if(HresultS[Hinshi][test_author][main_author] != None):
               m.printFixedLength(round(HresultS[Hinshi][test_author][main_author],4),10,endword=",",fileobj=outS)
           else:
               m.printFixedLength(HresultS[Hinshi][test_author][main_author],10,endword=",",fileobj=outS)
        m.printFixedLength(t_vocab_num[Hinshi][main_author][test_author],12,endword=",",fileobj=outS)
        for main_author in main_authors:
            m.printFixedLength(lt_and_vocab_num[Hinshi][main_author][test_author],6,endword=",",fileobj=outS)
        print("",file=outS)
    m.printFixedLength("L_vocab_num",11,endword=":,",fileobj=outS)
    # 最終行各メイン著者の学習作品の語彙数
    for main_author in main_authors:
        m.printFixedLength(l_vocab_num[Hinshi][main_author],11,endword=",",fileobj=outS)
    print("",file=outS)
    print("",file=outS)
outS.close



##  Dice係数の計算結果の出力
# 出力ファイルの指定
m.resetFile("../result/dice/"+m.nameFile(len(authors),dic,"Dice",cate,"#","#","#",sep,"#","#",".csv"))
outD = open("../result/dice/"+m.nameFile(len(authors),dic,"Dice",cate,"#","#","#",sep,"#","#",".csv"),"a")
# 全語彙（重複なし）での計算結果出力
print("All vocablalies",file=outD)
print("          ",end=",",file=outD)
for author in main_authors:
    m.printFixedLength(author,10,endword=",",fileobj=outD)
m.printFixedLength("T_vocab_num",12,endword=",",fileobj=outD)
m.printFixedLength("LT_and_vocab_num",16,endword=",",fileobj=outD)
print("",file=outD)    
for test_author in authors:
    m.printFixedLength(test_author,10,endword=":,",fileobj=outD)
    for main_author in main_authors:
        if(resultD[test_author][main_author] != None):
            m.printFixedLength(round(resultD[test_author][main_author],4),10,endword=",",fileobj=outD)
        else:
            m.printFixedLength(resultD[test_author][main_author],10,endword=",",fileobj=outD)
    m.printFixedLength(t_vocab_num["ALL"][main_author][test_author],12,endword=",",fileobj=outD)
    for main_author in main_authors:
            m.printFixedLength(lt_and_vocab_num["ALL"][main_author][test_author],6,endword=",",fileobj=outD)
    print("",file=outD)
m.printFixedLength("L_vocab_num",11,endword=":,",fileobj=outD)
for main_author in main_authors:
    m.printFixedLength(l_vocab_num["ALL"][main_author],11,endword=",",fileobj=outD)
print("",file=outD)
print("",file=outD)
# 各品詞毎での計算結果出力
for Hinshi in Hinshis:
    print(Hinshi+" vocablalies",file=outD)
    print("          ",end=",",file=outD)
    for author in main_authors:
        m.printFixedLength(author,10,endword=",",fileobj=outD)
    m.printFixedLength("T_vocab_num",12,endword=",",fileobj=outD)
    m.printFixedLength("LT_and_vocab_num",16,endword=",",fileobj=outD)
    print("",file=outD)
    for test_author in authors:
        m.printFixedLength(test_author,10,endword=":,",fileobj=outD)
        for main_author in main_authors:
           if(HresultD[Hinshi][test_author][main_author] != None):
               m.printFixedLength(round(HresultD[Hinshi][test_author][main_author],4),10,endword=",",fileobj=outD)
           else:
               m.printFixedLength(HresultD[Hinshi][test_author][main_author],10,endword=",",fileobj=outD)
        m.printFixedLength(t_vocab_num[Hinshi][main_author][test_author],12,endword=",",fileobj=outD)
        for main_author in main_authors:
            m.printFixedLength(lt_and_vocab_num[Hinshi][main_author][test_author],6,endword=",",fileobj=outD)
        print("",file=outD)
    m.printFixedLength("L_vocab_num",11,endword=":,",fileobj=outD)
    for main_author in main_authors:
        m.printFixedLength(l_vocab_num[Hinshi][main_author],11,endword=",",fileobj=outD)
    print("",file=outD)
    print("",file=outD)
