import sys
import MeCab
import glob

if len(sys.argv) == 2:
  dic = sys.argv[1]
  if dic == 'iNeologd':
    m = MeCab.Tagger ('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/')
  if dic == 'uNeologd':
    m = MeCab.Tagger ('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-unidic-neologd/')
  elif dic == 'Juman':
    m = MeCab.Tagger ('-Owakati -d /var/lib/mecab/dic/juman-utf8/')
  elif dic == 'Naist':
    m = MeCab.Tagger ('-Owakati -d /var/lib//mecab/dic/naist-jdic/')
  elif dic == 'Unidic':
    m = MeCab.Tagger ('-Owakati -d /var/lib/mecab/dic/unidic/')
  else:
    print("[ERROR] dictinary name is Wrong")
    print('expect "iNeologd uNeologd Juman Naist Unidic"')
    exit(1)
else:
  dic = ""
  m = MeCab.Tagger ('-Owakati')

#file_list = glob.glob('../data/*/*.txt-utf8-remove')
file_list = glob.glob('../data/*/*2/*.txt-utf8-remove2')


for file in file_list:
  words=""
  for line in open(file, 'r'):
    words += m.parse(line)
  #https://akamist.com/blog/archives/2656　ほぼコピペ
  output = file+"-wakati"+dic
  with open(output, 'w') as f:
    print(words[:-2], file=f)
  #https://techacademy.jp/magazine/2115　サンプル
  


print("done")
