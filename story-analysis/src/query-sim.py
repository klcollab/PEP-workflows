"""
Embed a query and retrieve all the stories that are similar to that query.
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-st', '--sim_thresh', type=float, default=0.95, required=False, help='Similarity Threshold.')
parser.add_argument('-tf','--themes', type=str, required=True, help='CSV of qualities that may be present in a story. E.g. a CSV of themes derived from the TA workflow.')
parser.add_argument('stories', nargs='+', help='Text files.')
args = parser.parse_args()

import os
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def read_file(fn):
  with open(fn,'r') as f: return(f.read())

storynames = []
stories = []
for storyfile in args.stories:
  storynames.append(Path(storyfile).stem)
  story = read_file(storyfile)
  stories.append(story)
storydf = pd.DataFrame(dtype='string')
storydf['Files'] = storynames
storydf['Stories'] = stories

#modelname = 'Alibaba-NLP/gte-base-en-v1.5'
modelname = 'Qwen/Qwen3-Embedding-8B'
#embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cpu'})
embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cuda'})

vector = FAISS.from_documents([Document(x) for x in stories], embeddings, normalize_L2=True)

querydf = pd.read_csv(args.themes)
for ndx,row in querydf.iterrows():
  simstories = vector.similarity_search(row.iloc[0],k=vector.index.ntotal,score_threshold=args.sim_thresh)
  matches = []
  for simstory in simstories:
    txt = simstory.page_content
    match = storydf[storydf['Stories'] == txt]
    matches.append(list(match['Files'])[0])
  print('Query '+str(ndx),end=': ')
  print(matches)
