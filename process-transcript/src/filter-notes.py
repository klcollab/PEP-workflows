import json
import warnings
from datetime import datetime
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def main(notes_files):
  notes = []
  for filen in notes_files:
    ndf = pd.read_csv(filen)
    print('Reading '+str(len(ndf['Note']))+' notes from '+filen)
    [notes.append(x) for x in ndf['Note']]

  # pick out the unique notes
  embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5", model_kwargs={'trust_remote_code': True, 'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
  vector = FAISS.from_documents([Document(x) for x in notes], embeddings, normalize_L2=True)
  total_notes = len(notes)
  SIM_THRESH = 0.75
  keep = []
  while (len(notes) > 0):
    note = notes.pop(0)
    if note in keep: continue
    if len(notes) <= 0: break # we're done if we run out of notes
    simdocs = vector.similarity_search(note,k=vector.index.ntotal,score_threshold=SIM_THRESH)
    keep.append(note)
    # log similars in the kill list just in case we want to vector.delete() later
    kill = []
    for doc in simdocs: kill.append(doc.page_content)
    print('DEBUG - Removing '+str(len(kill))+'/'+str(len(notes))+' docs.')
    if len(kill) > 0:
      notes = list(set(notes)-set(kill))
    print('DEBUG - '+str(len(notes))+' remain.')
      
  print('Done: '+str(len(keep))+'/'+str(total_notes)+' notes retained.')
  keepdf = pd.DataFrame(keep)
  keepdf.columns = ['Note']
  now = datetime.now()
  keepdf.to_csv('notes-sim-'+str(SIM_THRESH)+'-filtered-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+', help='CSV files containing notes.')
    args = parser.parse_args()
  main(args.args)
