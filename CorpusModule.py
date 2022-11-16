import numpy as np
import matplotlib.pyplot as plt
import glob
from gensim.models import word2vec
import sys
import csv
import os
import shutil 
from ClassWorks import Works

author = "Akutagawa"


def getWorkObj(author):

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
        works[index].setFilePath()
        index=index+1
    
    return works

### main ###
if __name__ == '__main__': #testModule.pyを実行すると以下が実行される（モジュールとして読み込んだ場合は実行されない）
    works = getWorkObj(author)
    for i in range(0,2):
        works[i].PrintSelfInformation()