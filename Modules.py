from curses import flash
import sys
import glob
import shutil
import re
import numpy as np
import csv 
import os
from ClassWorks import Works
import inspect

#著者名リスト
authors = ['ALL','Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
#辞書名リスト
dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']

###命名規則の関数------------------------------------------------------------------------------------
def nameFile(author:str,dic:str,what:str,target,vec,win,epoc,sep,model:str,other,extension:str):
    """ author : 著者名 , 
    dic : 辞書名 , 
    what : 何のファイルか , 
    target : 対象（Learn/Test/ALL）, 
    vec : モデルの学習次元数 , 
    win : モデルの学習窓数 , 
    epoc : モデルの学習回数 , 
    sep : 学習ファイルと検証ファイルの分割方法(サイズ),
    model :（Learn/ALL）を学習したモデル , 
    other : その他 , 
    extension : 拡張子 , 
    不要なパラメータは「#」とする """
    
    #入力エラーチェック
    err_flg = 0
    if(type(author)==int):
        author=str(author)
    elif((author in authors)==False):
        print("author name "+author+" is not found")
        continueOrExit()
    if( ((dic in dics)==False) and (dic != "#") ):
        print("dic name is not found")
        err_flg = 1
    if(extension[0]!="."):
        extension = "." + extension
    if(err_flg==1):
        print("@function 'nameFile' , Module.py")
        exit()
    
   
    #ファイル名の整形
    if(model=="#"):
        parameters = [what,"t"+str(target),"v"+str(vec),"w"+str(win),"e"+str(epoc),"s"+str(sep),model,other]
    else:
        parameters = [what,"t"+str(target),"v"+str(vec),"w"+str(win),"e"+str(epoc),"s"+str(sep),"m"+model,other]
    name = author + dic
    for param in parameters:
        name += ("-" + str(param))
    name += extension
    
    return name


###ファイルの初期化関数--------------------------------------------------------------------------------
def resetFile(filename : str):
    f = open(filename,"w")
    print("",file=f,end="")
    f.close


###コンティニュー確認-----------------------------------------------------------------------------------
def continueOrExit():
    while 1:
        print("continue ? ( y / n )")
        terminal_input = input()
        if( terminal_input == "n" ):
            exit()
        elif( terminal_input == "y" ):
            break
        else:
            print("please input 'y' or 'n'")


###csvファイル書き込み自作関数----------------------------------------------------------------------------------
def csvWriting(filename:str,item_list:list):
    f = open(filename,"a")
    item_num = len(item_list)
    for i in range(0,item_num):
        print(item_list[i],end="",file=f)
        if(i==(item_num-1)):
            print("",file=f)
        else:
            print(",",end="",file=f)
    f.close


###学習ファイルと検証ファイルの分割をリセットして著者名ディレクトリに戻す関数----------------------------------------------------------------------------------
def resetParse(dic):
    if((dic in dics)==False):
        print("[ERROR] input dic name is wrong @resetParse function")
        exit
    print("this function reset all Test or Learn dir, for "+dic)
    continueOrExit()

    for author in authors:
        for target in ["Test","Learn"]:
            print("glob "+author+"'s "+target+" files")            
            file_list = glob.glob("../data/"+author+"/"+target+"/*"+dic)
            for file_name in file_list:
                shutil.move( file_name , '../data/'+author+'/')


###ファイル読み出し関数----------------------------------------------------------------------------------
def readFile(author:str,dic:str,what:str,target:str,vec,win,epoc,sep,model:str,other,extension:str,dirpass):
    '''
    dirpass は「/」含める。ex.) "../result/mahala/"
    '''
    file_name=nameFile(author,dic,what,target,vec,win,epoc,sep,model,other,extension)
    print("readFile : "+dirpass+file_name)
    file = open(dirpass+file_name)
    data = file.read()
    return data

def readFileName(file_name,extension:str,dirpass):
    '''
    dirpass は「/」含める。ex.) "../result/mahala/"
    '''
    print("readFile : "+dirpass+file_name)
    file = open(dirpass+file_name)
    data = file.read()
    return data

##csvファイルを読み出して2重リスト化する
def csvToDoubleList(filename :str):
    csv_file = open(filename,"r")
    data = csv_file.read()
    double_list = data.splitlines()
    for i_csvToDoubleList in range(0,len(double_list)):
        double_list[i_csvToDoubleList] = double_list[i_csvToDoubleList].replace(","," ").split()
    csv_file.close
    return double_list

def idInsertToNovelList(author:str):
        '''
        author_worklist_novel.csvの3列目にworklist.csvから検索したIDを挿入する
        '''
        #ファイル名の定義
        worklist_filename = "../data/"+author+"/"+author+"_worklist.csv"
        worklist_novel_filename = "../data/"+author+"/"+author+"_worklist_novel.csv"
        #作品リストファイル.csvを2重リスト化して取得
        worklist_list = csvToDoubleList(worklist_filename)
        worklist_novel_list = csvToDoubleList(worklist_novel_filename)    

        #リストから1行ずつ取り出し
        for i in range(0,len(worklist_novel_list)):
            line = worklist_novel_list[i]
            work_name = line[0]
            kana = line[1] 

            #小説リストのi行目の3列目が無いなら
            if(len(line)==2):
                #作品名と仮名遣いの検索
                flag=0
                for j in range(0,len(worklist_list)):
                    if(worklist_list[j][0]==work_name and worklist_list[j][1]==kana):
                        id = worklist_list[j][2]
                        flag += 1
                if(flag >= 2):
                    print("[Worning] work:"+work_name+","+kana+" is duplicate.")
                    print("flag = "+str(flag))
                elif(flag==0):
                    print("[ERROR] work:"+work_name+","+kana+" is notfound.")
                else:
                    worklist_novel_list[i].insert(2,id)


        output_file = open("../data/"+author+"/"+author+"_novellist.csv","w")
        for i in range(0,len(worklist_novel_list)):
            csvWriting("../data/"+author+"/"+author+"_novellist.csv",worklist_novel_list[i])
        output_file.close

def get_id_list(file_obj):
    import csv
    csv_obj = csv.reader(file_obj,delimiter=",")
    id_list = []
    for line in csv_obj:
        id = line[2]
        id_list.append(id)
    return id_list

def get_file_list_none_novel(author,dic):
    #小説リストと作品リストのfileよみこみ
    novel_id_file = open("../data/"+author+"/"+author+"_novellist.csv","r")
    work_id_file = open("../data/"+author+"/"+author+"_worklist.csv","r")
    #idリストの生成
    novel_id_list = get_id_list(novel_id_file)
    work_id_list = get_id_list(work_id_file)
    novel_id_file.close
    work_id_file.close
    #作品idリストから小説作品のidを除外
    for novel_id in novel_id_list:
        try:
            work_id_list.remove(novel_id)
        except:
            print("id:"+novel_id+" is not in "+author+"_worklist")
    #テスト著者の小説以の作品読み込み
    file_list = []
    for work_id in work_id_list:
        file_list.extend(glob.glob('../lcorpus/'+work_id+'_*.txt-utf8-remove-wakati'+dic))
        if(len(glob.glob('../lcorpus/'+work_id+'_*.txt-utf8-remove-wakati'+dic))>1):
            print("[Warning]id="+str(work_id)+"'s works double")
            print(glob.glob('../lcorpus/'+work_id+'_*.txt-utf8-remove-wakati'+dic))
        if(len(glob.glob('../lcorpus/'+work_id+'_*.txt-utf8-remove-wakati'+dic))==0):
            print("[Warning]id="+str(work_id)+"'s work not found")
    print(len(file_list))
    return file_list



def openDatabaseCsv(author):
    #database.csvの読み込み
    database = open("../data/"+author+"/"+author+"_database.csv","r")
    csv_obj = csv.reader(database,delimiter=",")
    database.close
    return csv_obj


def getWorkObj(author,index_0,dir=""):
    database_obj = openDatabaseCsv(author)
    #作品インスタンスの生成
    works = {}
    index = index_0
    for line in database_obj:
        works[index] = Works()
        works[index].setSelfInformation(author,line,index)
        works[index].setFilePath(dir)
        index=index+1
    return works,index

def getFileNameByIdUseGlob(author,dic,work,file_list,dirnum=""):
    if(work.novel==True):
        dir = "novel"
    elif(work.novel==False):
        dir = "other"

    file_name=glob.glob("../data/"+author+"/"+dir+str(dirnum)+"/"+str(work.id)+"_*.txt*"+dic)
    if(len(file_name)==1):
        file_name=file_name[0]
    else:
        print(location())()
        print("file duplicate or lost")
        work.PrintSelfInformation()
        exit()
    file_list.append(file_name)
    return file_list

def location(depth=0):
    import inspect
    import os
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno

def printLocation(depth=0):
    import inspect
    import os
    frame = inspect.currentframe().f_back
    print(os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno)

def getWordListFromFileName(filename):
    file = open(filename)
    data = file.read()
    #ファイル内で使用されている単語を1次元リスト化
    wordlist = data.replace("\n"," ").split()
    if len(wordlist) == 0:
        print("[WARNING] File("+filename+") 's word_list length is Zero")
    return wordlist

def get_f_ave(file_name,model,vector):
    #初期化
    f_ave = np.zeros(vector)
    vocab_vec = np.zeros((vector), float)
    none = 0
    yes = 0
    #作品内で使用されている単語を足し合わせる
    word_list = getWordListFromFileName(file_name)
    for word in word_list:
        try:     
            vocab_vec = np.array(model.wv[word]) 
            yes += 1
        except:
            none += 1
            print("[WARNING] File("+file_name+") Word("+word+") 's vector is Zero")
            exit()
        f_ave += vocab_vec
    #作品の重心（特徴f(A,α)）
    f_ave /= (len(word_list) - none)
    return f_ave

def printFixedLength(val,length,seprate=" ",endword="\n",fileobj=None,flushval=False):
    ''''
        引数 length で指定した長さを固定長として出力する
        文字列型か数字型を渡すことを想定している
    '''
    val = str(val)
    if( int(length) < len(val) ):
        print("================================\n[Warning] ",location(),sep="")
        print("length smaller than len(val)\n================================")
        return
    print(val,end="",file=fileobj,flush=flushval)
    for i in range(0,int(length)-len(val)):
        print(" ",end="",file=fileobj,flush=flushval)
    print("",end=endword,file=fileobj,flush=flushval)

def printL(val:str ,seprate=" ",endword="\n",fileobj=None,flushval=False):
    print(len(val),sep=seprate,end=endword,file=fileobj,flush=flushval)

def save_ave(f:object,author,file_name,f_ave,output_file):
    print(author,file=output_file,end=",")
    f.write(' '.join(list(map(str,f_ave))))
    print(",",file=f,end="")
    print(file_name,file=f)

def getFileNameFromTargetListFile(fileobj,test_authors):
    ##  result/mahala/targetファイルから学習作品、検証作品のファイル名リストを生成
    reader = csv.reader(fileobj)
    next = 0
    ##  学習作品を求める
    for row in reader:
        if(row[0]=='main_lerans_index_list'):
            next = 1
        elif(next==1):
            L_file_names = row
            next = 0
            break

    ##  検証作品を求める
    T_file_names = {}       
    auth = 0
    for row in reader:
        if(len(test_authors)<=auth):
            return L_file_names,T_file_names   
        if(row[0]=='test_index_list['+test_authors[auth]+']'):
            next = 1
        elif(next==1):
            T_file_names[test_authors[auth]] = row
            next = 0
            auth += 1
    
    return 1

def getIdFromTargetListFile(fileobj,test_authors):
    ##  getFileNameFromTargetListFileと出力形式も違うから注意
    ##  result/mahala/targetファイルから学習作品、検証作品のIDリストを生成
    reader = csv.reader(fileobj)
    next = 0
    
    ##  学習作品を求める
    for row in reader:
        if(row[0]=='main_lerans_index_list'):
            next = 1
        elif(next==1):
            next = 2
        elif(next==2):
            L_file_Ids = row
            next = 0
            break
    
    ##  検証作品を求める
    T_file_Ids = []   
    auth = 0
    for row in reader:
        
        if(row[0]=='test_index_list['+test_authors[auth]+']'):
            next = 1
        elif(next==1):
            next = 2
        elif(next==2):
            T_file_Ids.extend(row)
            next = 0
            auth += 1
        if(len(test_authors)<=auth):
            return L_file_Ids,T_file_Ids

def printSizeUnit(obj:object,unit=None):
    byte = obj.__sizeof__()
    if(unit==None):
        denomi=1.0
    elif(unit=="k"or  unit=="K"):
        denomi=1000.0
    elif(unit=="m" or unit=="M"):
        denomi=1000000.0
    elif(unit=="g" or unit=="G"):
        denomi=1000000000.0
    byte = float(byte)/denomi
    print(type(obj),byte,str(unit)+"B")





### main ###
if __name__ == '__main__': #testModule.pyを実行すると以下が実行される（モジュールとして読み込んだ場合は実行されない）
    a=1000.0
    #sizeofUnit(a,"k")