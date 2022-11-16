import glob
import csv
import numpy as np 
from ClassWorks import Works

#著者名リスト
authors = ['Akutagawa','Arisima','Kajii','Kikuchi','Sakaguchi','Dazai','Nakajima','Natsume','Makino','Miyazawa']
dirs = ['novel','other']
author = "Makino"
dic = "uNeologd"
dir = "other"

for author in authors:
    for dir in dirs:
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

        file_list = glob.glob('../data/'+author+dir+'/*.txt-utf8-remove-wakatiuNeologd')


        correct_file_list=[]
        for i in range(0,len(works)):
            #works[i].PrintSelfInformation()
            id = works[i].id
            
            #小説かそうでないか
            if(dir == "other" and works[i].novel==True):
                continue
            elif(dir == "novel" and works[i].novel==False):
                continue
            else:
                pass

            #　data/author/dir/　内にファイルが1つだけあるかどうか
            ## idのファイルをglobでとる
            file_name = "../data/"+author+"/"+dir+"/"+str(id)+"_*"+dic
            file_name = glob.glob(file_name)
            ## file_nameが1つなら ok
            if(len(file_name)==1):
                file_name = file_name[0]
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
            correct_file_list.append(file_name)

        #print(correct_file_list)

        for real_file in file_list:
            if(real_file in correct_file_list):
                pass
            else:
                print(real_file)