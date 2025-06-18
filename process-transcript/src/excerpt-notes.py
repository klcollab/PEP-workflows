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
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

def remove_markup(llmtxt):
  if re.search('```*', llmtxt): # check for markup from the LLM
    clean = ''
    for line in llmtxt.split('\n'):
      if line.startswith('```'): continue
      clean += line
    llmtxt = clean.strip()
  return llmtxt

def remove_reason(llmtxt):
  loc = re.search('<think>', llmtxt)
  if loc != None:
    llmtxt = re.sub(r'<think>.*?</think>','',llmtxt,flags=re.DOTALL).strip()
  return llmtxt

SIM_THRESH = 0.95
def filter_notes(ndf,thresh):
    # pick out the unique notes
    embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5", model_kwargs={'trust_remote_code': True, 'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector = FAISS.from_documents([Document(x) for x in ndf['Note']], embeddings, normalize_L2=True)
    total_notes = vector.index.ntotal
    keep = []
    for note in ndf['Note']:
      if note in keep: continue
      if vector.index.ntotal <= 0: break # we're done if we run out of notes
      #simdocs = vector.similarity_search_with_relevance_scores(note,k=vector.index.ntotal,score_threshold=thresh)
      simdocs = vector.similarity_search(note,k=vector.index.ntotal,score_threshold=thresh)
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
    return keepdf
  
def main(txtfn, notespromptfn, run_filter):
  with open(txtfn,'r') as txtf: txt = txtf.read()
  excerpts = splitter.split_text(txt)
  with open(notespromptfn,'r') as ptf: notespt = ptf.read()
  # ignore 
  #modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:latest', 'qwen3:32b-fp16', 'llama4:17b-scout-16e-instruct-q8_0', 'llama4:16x17b']
  modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:latest', 'llama4:16x17b']
  #modelnames = ['tulu3:70b', 'phi4:latest', 'llama4:16x17b']
  #modelnames = ['llama3.3:70b-instruct-q8_0']
  #modelnames = ['llama4:16x17b']
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
      #print('DEBUG - These notes: \n'+notes_s)
      if (
          not re.search('[Nn]o content present.',notes_s) and
          not re.search('[Nn]o relevant',notes_s) and
          not re.search('[Nn]o mention',notes_s) and
          not re.search('[Dd]oes not mention',notes_s) and
          not re.search('[Dd]oesn\'t mention',notes_s) and
          not re.search('[Nn]o specific mention',notes_s) and
          not re.search('[Nn]o direct mention',notes_s) and
          not re.search('[Nn]o explicit mention',notes_s) and
          not re.search('[Nn]othing mentioned',notes_s) and
          not re.search('[Nn]o relevant content',notes_s) and
          not re.search('[Nn]o discussion',notes_s) and
          not re.search('[Dd]oes not discuss',notes_s) and
          not re.search('[Dd]oesn\'t discuss',notes_s) and
          not re.search('[Nn]o direct discussion',notes_s)
        ):
        notes_s = remove_markup(notes_s)
        notes_s = remove_reason(notes_s)

        try: excerpt_notes = json.loads(notes_s)
        except json.decoder.JSONDecodeError:
          print('\nDEBUG - These notes: \n'+notes_s)
          except_prompt = PromptTemplate.from_template("""If the following JSON is not properly 
formatted, reformat it. Do not otherwise comment. Just respond with the properly formatted JSON.
<json>{json}</json>.
"""
                                                       )
          except_chain_model = except_prompt | model | StrOutputParser()
          reformatted = except_chain_model.invoke({'json': notes_s}).strip()
          reformatted = remove_markup(reformatted)
          reformatted = remove_reason(reformatted)
          print('Reformatted\n'+reformatted)
          excerpt_notes = json.loads(reformatted)
          
        note_list += excerpt_notes
        
    notesdf = pd.DataFrame(note_list)
    notesdf.columns = ['Note']

    now = datetime.now()
    if (run_filter):
      keepdf = filter_notes(notesdf,SIM_THRESH)
      keepdf.to_csv('notes-'+'sim-'+str(SIM_THRESH)+'-'+'-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)
    else:
      print('Pulled '+str(len(notesdf['Note']))+' from '+str(len(excerpts))+' excerpts.')
      notesdf.to_csv('notes-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', '--transcript', type=str, default='none', required=True, help='Interview transcript.')
    parser.add_argument('-np', '--notes_prompt', type=str, default='none', required=True, help='Template format prompt for extracting notes.')
    parser.add_argument('-fn', '--filter_notes', action='store_true', required=False, help='Whether or not to filter the notes by similarity.')
    args = parser.parse_args()
  main(args.transcript,args.notes_prompt,args.filter_notes)
