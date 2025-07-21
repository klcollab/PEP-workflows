import re
import json
import warnings
from datetime import datetime
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

modelnames = ['llama3.3:70b-instruct-q8_0', 'phi4:latest', 'llama4:16x17b']
#modelnames = ['zephyr:latest']
#modelnames = ['llama4:16x17b']
splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-story', '--story', type=str, default='none', required=True, help='Prior version of the story.')
parser.add_argument('-tran', '--transcript', type=str, default='none', required=True, help='Interview transcript.')
parser.add_argument('-np', '--notes_prompt', type=str, default='none', required=True, help='Template format prompt for extracting notes.')
args = parser.parse_args()

with open(args.story,'r') as f: story = f.read()
with open(args.transcript,'r') as f: tran = f.read()
with open(args.notes_prompt,'r') as f: np = f.read()

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

def remove_envelope_text(llmtxt):
  m = re.search('\[', llmtxt)
  if m != 'None': llmtxt = llmtxt[m.start(0):]
  m = re.search('\]', llmtxt)
  if m != 'None': llmtxt = llmtxt[:m.end(0)]
  return llmtxt

def skipit(note):
    return (
      re.search('[Nn]o content present.',notes_s) or
      re.search('[Nn]o relevant',notes_s) or
      re.search('[Nn]o mention',notes_s) or
      re.search('[Dd]oes not mention',notes_s) or
      re.search('[Dd]oesn\'t mention',notes_s) or
      re.search('[Nn]o specific mention',notes_s) or
      re.search('[Nn]o direct mention',notes_s) or
      re.search('[Nn]o explicit mention',notes_s) or
      re.search('[Nn]othing mentioned',notes_s) or
      re.search('[Nn]o relevant content',notes_s) or
      re.search('[Nn]o discussion',notes_s) or
      re.search('[Dd]oes not discuss',notes_s) or
      re.search('[Dd]oesn\'t discuss',notes_s) or
      re.search('[Nn]o direct discussion',notes_s)
    )

excerpts = splitter.split_text(tran)

now = datetime.now()
for modelname in modelnames:
  model = OllamaLLM(model=modelname, temperature=0.0, num_predict=-1)
  prompt = PromptTemplate.from_template(np)
  notes_chain_model = prompt | model | StrOutputParser()
  note_list = []
  excerptnum = 0
  for excerpt in excerpts:
    excerptnum += 1
    print(modelname+' is taking notes on excerpt '+str(excerptnum)+'/'+str(len(excerpts))+'\r',end='')
    notes_s = notes_chain_model.invoke({'story': story, 'excerpt': excerpt}).strip()
    if (not skipit(notes_s)):
      notes_s = remove_markup(notes_s)
      notes_s = remove_reason(notes_s)

      try: excerpt_notes = json.loads(notes_s)
      except json.decoder.JSONDecodeError:
        print('\nDEBUG - These notes: \n'+notes_s)
        except_prompt = PromptTemplate.from_template("""If the
following string is not properly formatted as a Python list, reformat
it. Use a flat list of string, one note per per array element. Do not
otherwise comment. Just respond with the properly formatted list.
<python_list>{python_list}</python_list>.
""")
        except_chain_model = except_prompt | model | StrOutputParser()
        reformatted = except_chain_model.invoke({'python_list': notes_s}).strip()
        reformatted = remove_markup(reformatted)
        reformatted = remove_reason(reformatted)
        reformatted = remove_envelope_text(reformatted)
        print('Reformatted\n'+reformatted)
        excerpt_notes = json.loads(reformatted)

      note_list += excerpt_notes

  notesdf = pd.DataFrame(note_list)
  notesdf.columns = ['Note']
    
  print('Pulled '+str(len(notesdf['Note']))+' notes from '+str(len(excerpts))+' excerpts.')
  notesdf.to_csv('delta-notes-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False)
