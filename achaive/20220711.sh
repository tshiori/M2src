#alpha_val=args[1]
#min_alpha_val=args[2]
#seed_val=args[3]
#workers_val=args[4]
#vector=int(args[5])
PYTHONHASHSEED=1 python3 ALLCorpusMakeModel.py 0.01 0.0001 1 1 200
PYTHONHASHSEED=1 python3 ALLCorpusMakeModel.py 0.005 0.00005 1 1 100
PYTHONHASHSEED=1 python3 ALLCorpusMakeAuthorModel.py 0.01 0.0001 1 1 200
PYTHONHASHSEED=1 python3 ALLCorpusMakeAuthorModel.py 0.005 0.00005 1 1 100
PYTHONHASHSEED=1 python3 PCA.py 200
PYTHONHASHSEED=1 python3 PCA.py 100


