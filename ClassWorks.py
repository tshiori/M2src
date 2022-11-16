import numpy as np
import glob
import os
import inspect


class Works:
    authors = []
    category_book = {911:"poetry",912:"play",913:"novel",914:"essay",915:"diary",916:"record",917:"Proverbs",918:"works",919:"chinese writing"}
    
    def __init__(self):
        self.index=""
        self.author=""
        self.work_name=""
        self.pseudonym=""
        self.id=""
        self.novel=""
        self.duplicate=""
        self.category=""
        self.subcategory=""
        self.filepath=""
        self.feature=[]
        self.mahala_dist=0
      
    def setSelfInformation(self,author,line,index):
        '''
        著者名とデータベースから読み込んだ1行をリスト形式にして読み込む
        '''
        try:
            self.index=index
            self.author=author
            self.work_name=line[0]
            self.pseudonym=line[1]
            self.id=int(line[2])

            if line[3]=="novel":
                self.novel=True
            elif line[3]=="other":
                self.novel=False
            else:
                print(location())
                print("[Warning] "+author+" id:"+str(self.id)+"works novel or other is wrong in database")
                print(line[3])

            if line[4]=="duplicate":
                self.duplicate=True
            elif str(line[4])=="0":
                self.duplicate=False
            else:
                print(location())
                print("[Warning] "+author+" id:"+str(self.id)+"works duplicate is wrong in database")
                print(line[4])

            self.category=int(line[5])

            if (len(line)>=7 and line[6]):
                self.subcategory=int(line[6])

        except:

            print(str(line)+"except")

    def setFilePath(self,dir=""):
            if(self.novel==False):
                file_path = glob.glob("../data/"+self.author+"/other"+dir+"/"+str(self.id)+"_*uNeologd")
            elif(self.novel==True):
                file_path = glob.glob("../data/"+self.author+"/novel"+dir+"/"+str(self.id)+"_*uNeologd")
            
            if(len(file_path)==1):
                if(os.path.exists(file_path[0])):
                    self.filepath=file_path[0]
                else:
                    print(location())
                    print(id+"'s file not found")
            else:
                print(location())
                print("other len(file_path)=="+str(len(file_path)),end=",")
                print("id="+str(self.id))

    def PrintSelfInformation(self):
        print(str(self.index))
        print(self.author)
        print(self.work_name)
        print(self.pseudonym)
        print(str(self.id))
        print(self.novel)
        print(self.duplicate)
        print(str(self.category))
        if(self.subcategory):
            print(str(self.subcategory))
        else:
            print("none-sub category")
        print(self.filepath)

    def PrintSelfInformation_file(self,f):
        print(str(self.index),file=f)
        print(self.author,file=f)
        print(self.work_name,file=f)
        print(self.pseudonym,file=f)
        print(str(self.id),file=f)
        print(self.novel,file=f)
        print(self.duplicate,file=f)
        print(str(self.category),file=f)
        if(self.subcategory):
            print(str(self.subcategory),file=f)
        else:
            print("none-sub category",file=f)
        print(self.filepath,file=f)
      

def location(depth=0):
  frame = inspect.currentframe().f_back
  return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno

