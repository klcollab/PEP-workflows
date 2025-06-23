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

#modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:latest', 'qwen3:32b-fp16', 'llama4:17b-scout-16e-instruct-q8_0', 'llama4:16x17b']
modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:latest', 'llama4:16x17b']
#modelnames = ['tulu3:70b', 'phi4:latest', 'llama4:16x17b']
#modelnames = ['llama3.3:70b-instruct-q8_0']
#modelnames = ['llama4:16x17b']
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1400,
    chunk_overlap=75,
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
  
def main(txtfn, notespromptfn):
  with open(txtfn,'r') as txtf: txt = txtf.read()
  excerpts = splitter.split_text(txt)
  with open(notespromptfn,'r') as ptf: notespt = ptf.read()
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
    print('Pulled '+str(len(notesdf['Note']))+' notes from '+str(len(excerpts))+' excerpts.')
    notesdf.to_csv('notes-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', '--transcript', type=str, default='none', required=True, help='Interview transcript.')
    parser.add_argument('-np', '--notes_prompt', type=str, default='none', required=True, help='Template format prompt for extracting notes.')
    args = parser.parse_args()
  main(args.transcript,args.notes_prompt)
