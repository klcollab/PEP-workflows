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
  total_notes = vector.index.ntotal
  SIM_THRESH = 0.95
  keep = []
  for note in notes:
    if note in keep: continue
    if vector.index.ntotal <= 0: break # we're done if we run out of notes
    simdocs = vector.similarity_search(note,k=vector.index.ntotal,score_threshold=SIM_THRESH)
    keep.append(note)
    # log similars in the kill list just in case we want to vector.delete() later
    kill = []
    #for doc in [tup[0] for tup in simdocs]: kill.append([doc.id,doc.page_content])
    for doc in simdocs: kill.append([doc.id,doc.page_content])
    if len(kill) > 0:
      kill_ids = [tup[0] for tup in kill]
      if not vector.delete(kill_ids): warnings.Warn("Failed to delete entries from FAISS db.")

  print('Done: '+str(len(keep))+'/'+str(total_notes)+' notes retained.')
  keepdf = pd.DataFrame(keep)
  keepdf.columns = ['Note']
  now = datetime.now()
  keepdf.to_csv('notes-'+'filtered-sim-'+str(SIM_THRESH)+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+', help='CSV files containing notes.')
    args = parser.parse_args()
  main(args.args)
