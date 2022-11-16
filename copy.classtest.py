import sys
import numpy as np

class Estimate:
    main_authors = []
    
    def __init__(self):
        self.file_name=""
        self.correct_author=""
        self.mahala_dist={}
        self.result_min=""
        self.result_distri={}
        self.result_distri_val=""
        self.result_code={}
        self.result_match={}

    def estimateResultMin(self):
        min_dist=min(np.array(list(map(float,self.mahala_dist.values()))))
        for key,val in self.mahala_dist.items():
            if( str(min_dist) == str(val) ):
                self.result_min = key

    def estimateResultDistribution(self,border):
        if(type(border)==dict):
            for main_author in Estimate.main_authors:
                self.result_distri[main_author] =  main_author if (float(self.mahala_dist[main_author])<border[main_author]) else "other"
        else:
            for main_author in Estimate.main_authors:
                self.result_distri[main_author] =  main_author if (float(self.mahala_dist[main_author])<border) else "other"

    def calcResultCode(self):
        for main_author in self.main_authors:
            if( self.result_distri[main_author]==self.correct_author and  self.result_min==self.correct_author ):
                self.result_code[main_author] = 3
            elif( self.result_distri[main_author]==self.correct_author and  self.result_min!=self.correct_author ):
                self.result_code[main_author] = 2
            elif( self.result_distri[main_author]!=self.correct_author and  self.result_min==self.correct_author ):
                self.result_code[main_author] = 1
            elif( self.result_distri[main_author]!=self.correct_author and  self.result_min!=self.correct_author ):
                self.result_code[main_author] = 0

    def calcResultMatch(self):
        for main_author in Estimate.main_authors:
            self.result_match[main_author] = True if self.result_distri[main_author]==self.result_min else False

