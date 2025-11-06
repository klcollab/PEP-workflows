import pandas as pd
import numpy as np
from numpy.linalg import norm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

#modelname = 'Alibaba-NLP/gte-base-en-v1.5'
modelname = 'Qwen/Qwen3-Embedding-8B'

def csim(u,v):
  return(np.dot(u,v))
  
def read_file(fn):
  with open(fn,'r') as f: return(f.read())

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('reports', nargs='+', help='Text files.')
args = parser.parse_args()

prefix = Path(args.reports[0]).stem
prefix = prefix[0:prefix.find('-')]

if Path(prefix+'embn.csv').exists():
  embdf = pd.read_csv(prefix+'embn.csv',index_col=0)
  report_names = list(embdf.index)

# for each [new] record, get an embed
if 'report_names' not in locals(): report_names = []
embs = []
embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cpu'})
for report in args.reports:
  if 'embdf' in locals() and Path(report).stem in embdf.index: continue
  print('Embedding '+report)
  report_names.append(Path(report).stem)
  r = read_file(report)
  embs.append(embeddings.embed_query(r))

# put them in a matrix and normalize it
embs = np.array(embs)
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embn = embs/norms

if 'embdf' in locals():
  priorembn = np.array(embdf)
  embn = np.vstack([priorembn,embn])

embdf = pd.DataFrame(embn)
embdf.index = report_names
embdf.to_csv(prefix+'embn.csv')

if len(report_names) > 1:
  sims = []
  # get pairwise similarities
  for i in range(0,len(embn)):
    row = []
    for j in range(0,len(embn)):
      row.append(csim(embn[i],embn[j]))
    sims.append(row)

  sims = np.array(sims)

  simsdf = pd.DataFrame(sims)
  simsdf.columns = report_names
  simsdf.index = report_names
  simsdf.to_csv(prefix+'sims.csv')

  # report the worst matches
  mins = pd.concat([simsdf.idxmin(axis=1),simsdf.min(axis=1)],axis=1)
  mins.columns = ['id','score']
  mins.to_csv(prefix+'mins.csv')
  print('\nThe worst matches were:')
  print(mins.loc[mins['score'] == mins['score'].min()])

  # report the best matches
  nonself = simsdf[simsdf < 1.0-1e-5]
  maxs = pd.concat([nonself.idxmax(axis=1),nonself.max(axis=1)],axis=1)
  maxs.columns = ['id','score']
  maxs.to_csv(prefix+'maxs.csv')
  print('\nThe best matches were:')
  print(maxs.loc[maxs['score'] == maxs['score'].max()])
