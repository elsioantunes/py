#!/usr/bin/python
'''
    sudo apt-get update
    #sudo apt-get install python-sklearn
    pip install -U scikit-learn
    sudo rm -rf /tmp/*
    sudo apt-get install ncdu

'''
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import random

data, meta = arff.loadarff('krk_database.arff')
df = pd.DataFrame(data) 
X = df.iloc[:,0:6]
y = df.Classe


################################################################

def chebyshev(x1, y1, x2, y2):
  return max(abs((int)(x1-x2)),abs((int)(y1-y2)))

def treino(dados, classes, vizinhos=7, confusion=0, rnd=824):
  (x_train, x_test, y_train, y_test) = tts(dados, classes, test_size=.25, random_state=rnd)
  alg = KNeighborsClassifier(vizinhos)
  alg.fit(x_train, y_train);
  score = alg.score(x_test, y_test);
  if confusion:
    print (score)
    y_pred = alg.predict(x_test)
    return pd.DataFrame(confusion_matrix(y_test, y_pred))
  else:
    return score

'''
def descricao_de_empate(xBK, yBK, xWR, yWR, xWK, yWK):
  ataque = chebyshev(xBK, yBK, xWR, yWR) == 1
  naodefesa = chebyshev(xWK, yWK, xWR, yWR) > 1
  nocanto = xBK == 0 and yBK == 0
  afogamento = nocanto and yWR == 1 and ((xWK == 2 and yWK == 0) or xWR == 1)
  return naodefesa and ataque or afogamento

def descricao_de_mate(xBK, yBK, xWR, yWR, xWK, yWK):
  quina = (yBK == yWK) or (yBK < 2 and yWK == 1)
  wkCaso1 = quina and xBK == 0 and xWK == 2 and xWR == 0
  wkCaso2 = xBK == xWK and yBK == 0 and yWK == 2 and yWR == yBK
  return chebyshev(xBK, yBK, xWR, yWR) > 1 and (wkCaso1 or wkCaso2)

def descricao_de_L_pattern(xBK, yBK, xWR, yWR, xWK, yWK):
  chuta1 = xBK == xWR and yBK == yWK and abs(xBK - xWK) == 2
  chuta2 = yBK == yWR and xBK == xWK and abs(yBK - yWK) == 2
  return (chuta2 or chuta1) and chebyshev(xBK, yBK, xWR, yWR) > 1
'''


'''
dimensoes = ['xwk','ywk','xbk','ybk','xwr','ywr']
f = {i:[] for i in dimensoes}

for xWK, yWK, xWR, yWR, xBK, yBK in X.values:
  f['xwk'].append(xWK/5)
  f['ywk'].append(yWK/5)
  f['xbk'].append(xBK/7)
  f['ybk'].append(yBK/7)
  f['xwr'].append(xWR/5)
  f['ywr'].append(yWR/5)

x = pd.DataFrame(f)  
print (treino(X, y, confusion=1, rnd=0))
'''

graf = []
for i in range(150,350):
    score = ('{:.3f}'.format(treino(X, y, vizinhos=7, rnd=i)), i)
    graf.append(score)
    print (score, end=' ')
print (('0.750', 48))  
print (min(graf))
print (max(graf))



