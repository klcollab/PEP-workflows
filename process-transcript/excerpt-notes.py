import re
import json
import warnings
from datetime import datetime
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False,
)

def main(txtfn, notespromptfn):
  with open(txtfn,'r') as txtf: txt = txtf.read()
  excerpts = splitter.split_text(txt)
  with open(notespromptfn,'r') as ptf: notespt = ptf.read()
  # ignore 
  #modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:latest', 'qwen3:32b-fp16', 'llama4:17b-scout-16e-instruct-q8_0', 'llama4:16x17b']
  modelnames = ['llama3.3:70b-instruct-q8_0']
  for modelname in modelnames:
    print('Running using '+modelname)
    model = OllamaLLM(model=modelname, temperature=0.0, num_predict=-1)
    notes_prompt = PromptTemplate.from_template(notespt)
    notes_chain_model = notes_prompt | model | StrOutputParser()
    note_list = []
    excerptnum = 0
    for excerpt in excerpts:
      excerptnum += 1
      print('Processing excerpt '+str(excerptnum)+'/'+str(len(excerpts))+'\r',end='')
      notes_s = notes_chain_model.invoke({'excerpt': excerpt}).strip()
      if (
          not re.search('[Nn]o content present.',notes_s) and
          not re.search('[Nn]o mention of',notes_s) and
          not re.search('[Nn]o relevant content',notes_s) and
          not re.search('[Nn]o discussion of',notes_s)
        ):
        excerpt_notes = json.loads(notes_s)
        note_list += excerpt_notes
        
    notes_df = pd.DataFrame(note_list)
    notes_df.columns = ['Note']
  
    # pick out the unique notes
    embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5", model_kwargs={'trust_remote_code': True, 'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector = FAISS.from_documents([Document(x) for x in notes_df['Note']], embeddings, normalize_L2=True)
    total_notes = vector.index.ntotal
    SIM_THRESH = 0.95
    keep = []
    for note in notes_df['Note']:
      if note in keep: continue
      if vector.index.ntotal <= 0: break # we're done if we run out of notes
      #simdocs = vector.similarity_search_with_relevance_scores(note,k=vector.index.ntotal,score_threshold=SIM_THRESH)
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
    keepdf.to_csv('notes-'+'sim-'+str(SIM_THRESH)+'-'+'-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', '--transcript', type=str, default='none', required=True, help='Interview transcript.')
    parser.add_argument('-np', '--notes_prompt', type=str, default='none', required=True, help='Template format prompt for extracting notes.')
    args = parser.parse_args()
  main(args.transcript,args.notes_prompt)
