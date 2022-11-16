import csv
from ClassWorks import Works
import glob
import inspect
import os


def main():

    #author = "Kikuchi"
    #著者名リスト
    authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
    #辞書名リスト
    dics = ['Ipadic','Naist','iNeologd','uNeologd','Juman','Unidic']
    dic = "uNeologd"
    #ジャンル毎の作品数
    works_num_category={}

    for author in authors:
        print("############"+author+"############")
        #作品数のリストに著者名のキーを登録
        works_num_category[author] = {}

        #database.csvの読み込み
        database = open("../data/"+author+"/"+author+"_database.csv","r")
        csv_obj = csv.reader(database,delimiter=",")
        database.close

        #作品インスタンスの生成
        works = {}
        index = 0
        for line in csv_obj:
            works[index] = Works()
            works[index].setSelfInformation(author,line,index)
            index=index+1
        for i in range(0,len(works)):
            #works[i].PrintSelfInformation()
            id = works[i].id
            
            #重複ファイルかどうか
            if(works[i].duplicate==True):
                continue
            elif(works[i].duplicate==False):
                pass
            else:
                print(location())
                print(author+" id:"+str(works[i].id)+" works.duplicate is void")

            #小説かそうでないか
            if(works[i].novel==True):
                dir = "novel"
            elif(works[i].novel==False):
                dir = "other"
            else:
                dir = ""
                print(location())
                print(author+" id:"+str(works[i].id)+" works.novel is void")

            #　data/author/dir/　内にファイルが1つだけあるかどうか
            ## idのファイルをglobでとる
            file_name = "../data/"+author+"/"+dir+"/"+str(id)+"_*"+dic
            file_name = glob.glob(file_name)
            ## file_nameが1つなら ok
            if(len(file_name)==1):
                pass
            elif(len(file_name)>=2):#2つ以上
                print(author+" id:"+str(works[i].id)+" file duplicate  :  "+dir)
                print("file_name =",end=" ")
                print(file_name)
                continue
            elif(len(file_name)==0):#みつからない
                print(author+" id:"+str(works[i].id)+" file not found  :  "+dir)
                print("file_name =",end=" ")
                print(file_name)
                continue
            else:
                print("unexpected error")

            #カテゴリー毎にカウント
            try:
                works_num_category[author][works[i].category]+=1
            except:
                works_num_category[author][works[i].category]=1
        
    print("############ sum ############")

    for author in authors:
        print(author)
        print("    "+str(works_num_category[author]))
        work_sum=0
        for category in works_num_category[author]:
            work_sum+=works_num_category[author][category]
        print("    "+str(work_sum))

    
def location(depth=0):
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno

main()