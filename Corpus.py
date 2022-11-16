import numpy as np
import matplotlib.pyplot as plt
import glob
from gensim.models import word2vec
import sys
import csv
import os
import shutil 
from ClassWorks import Works

author = "Kikuchi"

database = open("../data/"+author+"/"+author+"_database.csv","r")
#database.csvの読み込み
csv_obj = csv.reader(database,delimiter=",")
database.close
#作品インスタンスの生成
works = {}
index = 0
for line in csv_obj:
    works[index] = Works()
    works[index].setSelfInformation(author,line,index)
    index=index+1

print(len(works))
cp_list_other = []
cp_list_novel = []
output = open("../debug/error-corpus-"+author,"w")
count=0
for i in range(0,len(works)):
    if(works[i].novel==False):
        file_name = glob.glob("../../MExperiment/lcorpus/"+str(works[i].id)+"_*uNeologd")
        if(len(file_name)==1):
            if(os.path.exists(file_name[0])):
                cp_list_other.append(file_name[0])
            else:
                print(id+"'s file not found")
        else:
            print("other len(file_name)=="+str(len(file_name)),end=",",file=output)
            print("id="+str(works[i].id),file=output)
            print("id="+str(works[i].id))
            count=count+1

    elif(works[i].novel==True):
        file_name = glob.glob("../../MExperiment/data/"+author+"/*/"+str(works[i].id)+"_*uNeologd")
        if(len(file_name)==1):
            if(os.path.exists(file_name[0])):
                cp_list_novel.append(file_name[0])
            else:
                print(id+"'s file not found")
        else:
            print("novel len(file_name)=="+str(len(file_name)),end=",",file=output)
            print("id="+str(works[i].id),file=output)
            print("id="+str(works[i].id))
            count=count+1

print(len(cp_list_novel),len(cp_list_other),count)
sum=len(cp_list_novel)+len(cp_list_other)+count
print(sum)

for file_name in cp_list_other:
    try:
        shutil.copy(file_name,"../data/"+author+"/other/")
    except:
        print("[Error] Copy"+file_name)

for file_name in cp_list_novel:
    try:
        shutil.copy(file_name,"../data/"+author+"/novel/")
    except:
        print("[Error] Copy"+file_name)

