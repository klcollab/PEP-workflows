import json
import warnings
from datetime import datetime
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def get_longest(docs):
  if not docs or len(docs) <= 0: return ''
  max_length = -1
  max_note = ''
  for doc in docs:
    if len(doc.page_content) > max_length:
      max_length = len(doc.page_content)
      max_note = doc.page_content
  return max_note

def model_from_filename(fn):
  pre = fn[0:fn.find(':')]
  return pre[pre.rfind('-')+1:]

def main(notes_files,st):
  #now = datetime.now() # get the clock early for crisper diffs
  input_id = '+'.join([model_from_filename(x) for x in notes_files])+':'
  notes = []
  for filen in notes_files:
    ndf = pd.read_csv(filen)
    print('Reading '+str(len(ndf['Note']))+' notes from '+filen)
    [notes.append(x) for x in ndf['Note']]

  notes = sorted(notes)
  # pick out the unique notes
  embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5", model_kwargs={'trust_remote_code': True, 'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
  vector = FAISS.from_documents([Document(x) for x in notes], embeddings, normalize_L2=True)
  total_notes = len(notes)
  keep = []
  while (len(notes) > 0):
    note = notes.pop(0)
    if note in keep: continue
    if len(notes) <= 0: break # we're done if we run out of notes
    simdocs = vector.similarity_search(note,k=vector.index.ntotal,score_threshold=st)
    longest = get_longest(simdocs)
    keep.append(longest)
    # log similars in the kill list just in case we want to vector.delete() later
    kill = []
    for doc in simdocs: kill.append(doc.page_content)
    #print('DEBUG - Removing '+str(len(kill))+'/'+str(len(notes))+' docs.')
    if len(kill) > 0:
      notes = list(set(notes)-set(kill))
    notes = sorted(notes)
    #print('DEBUG - '+str(len(notes))+' remain.')

  keep = sorted(set(keep))
  print('Done: '+str(len(keep))+'/'+str(total_notes)+' notes retained.')
  keepdf = pd.DataFrame(keep)
  keepdf.columns = ['Note']
  #keepdf.to_csv('notes-sim-'+str(st)+'-filtered-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)
  keepdf.to_csv('notes-sim-'+str(st)+'-filtered-'+input_id+'.csv', index=False)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-st', '--sim_thresh', type=float, default=0.95, required=False, help='Similarity Threshold below which 2 strings are considered similar.')
    parser.add_argument('args', nargs='+', help='CSV files containing notes.')
    args = parser.parse_args()
  main(args.args, args.sim_thresh)
