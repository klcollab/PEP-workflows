"""
Document Embedding and Similarity Analysis Tool

This script processes text documents to generate embeddings using HuggingFace models,
computes pairwise similarities between documents, and outputs analysis results in CSV format.

Usage:
    python embed-docs.py report1.txt report2.txt ...

Inputs:
    One or more text files containing documents to analyze

Outputs:
    - {prefix}embn.csv: Normalized embeddings for all documents
    - {prefix}sims.csv: Pairwise similarity matrix (if multiple documents)
    - {prefix}mins.csv: Worst matches (minimum similarities)
    - {prefix}maxs.csv: Best matches (maximum similarities)

Where {prefix} is derived from the first input file's stem (before the first hyphen).

The script handles incremental processing - if embeddings already exist for some documents,
it will only process new documents and merge results.
"""

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

# Check if embeddings already exist for previous documents
if Path(prefix+'embn.csv').exists():
  embdf = pd.read_csv(prefix+'embn.csv',index_col=0)
  report_names = list(embdf.index)

# Process each document to generate embeddings
# Skip documents that already have embeddings to enable incremental processing
if 'report_names' not in locals(): report_names = []
embs = []
embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cpu'})
for report in args.reports:
  if 'embdf' in locals() and Path(report).stem in embdf.index: continue
  print('Embedding '+report)
  report_names.append(Path(report).stem)
  r = read_file(report)
  embs.append(embeddings.embed_query(r))

# Convert embeddings to numpy array and normalize to unit vectors
# Normalization ensures cosine similarity is equivalent to dot product
embs = np.array(embs)
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embn = embs/norms

# Merge with existing embeddings if this is incremental processing
if 'embdf' in locals():
  priorembn = np.array(embdf)
  embn = np.vstack([priorembn,embn])

# Save normalized embeddings to CSV
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
