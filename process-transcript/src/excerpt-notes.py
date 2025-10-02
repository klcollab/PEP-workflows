import re
import json
import csv
import warnings
from datetime import datetime
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'olmo2:13b', 'phi4:14b', 'qwen3:32b-fp16', 'llama4:17b-scout-16e-instruct-q8_0', 'llama4:16x17b', 'wizardlm2:8x22b']
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

  #code_model = OllamaLLM(model='gpt-oss:20b', temperature=0.0, num_predict=-1)
  code_model = OllamaLLM(model='codestral:22b', temperature=0.0, num_predict=-1)
  code_prompt = PromptTemplate.from_template(
    """Reformat this list of bullets into a JSON array. Use a flat list
of strings, one note per per array element. Do not otherwise
comment. Just respond with the properly formatted list.
<bullets>{bullets}</bullets>.  
""")

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

        code_chain_model = code_prompt | code_model | StrOutputParser()
        notes_s = code_chain_model.invoke({'bullets': notes_s})
        notes_s = notes_s.encode('utf-8').decode('unicode_escape')

        try: excerpt_notes = json.loads(notes_s)
        except json.decoder.JSONDecodeError:
          print('\nDEBUG - These notes: \n'+notes_s)
          except_prompt = PromptTemplate.from_template(
            """If the following string is not properly formatted as a JSON array,
reformat it. Use a flat list of string, one note per per array
element. Do not otherwise comment. Just respond with the properly
formatted list. Pay close attention to quotes. If a list element starts with 
a double quote, it must end with a double quote.
<python_list>{python_list}</python_list>.
""")
          except_chain_model = except_prompt | code_model | StrOutputParser()
          reformatted = except_chain_model.invoke({'python_list': notes_s}).strip()
          reformatted = remove_markup(reformatted)
          reformatted = remove_reason(reformatted)
          print('Reformatted\n'+reformatted)
          try: excerpt_notes = json.loads(reformatted)
          except json.decoder.JSONDecodeError:
            print('\nDEBUG - Failed again: \n'+notes_s)
            except_prompt = PromptTemplate.from_template(
              """Correct this JSON syntax. Make sure every entry is quoted properly. 
              Do not comment on it. Merely correct it. Double check the quoting.
              <python_list>{python_list}</python_list>.
              """
            )
            except_chain_model = except_prompt | code_model | StrOutputParser()
            reformatted = except_chain_model.invoke({'python_list': notes_s}).strip()
            reformatted = remove_markup(reformatted)
            reformatted = remove_reason(reformatted)
            excerpt_notes = json.loads(reformatted)
        note_list += excerpt_notes
        
    notesdf = pd.DataFrame(note_list)
    notesdf.columns = ['Note']

    now = datetime.now()
    print('Pulled '+str(len(notesdf['Note']))+' notes from '+str(len(excerpts))+' excerpts.')
    notesdf.to_csv('notes-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index=False, quoting=csv.QUOTE_ALL)

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', '--transcript', type=str, default='none', required=True, help='Interview transcript.')
    parser.add_argument('-np', '--notes_prompt', type=str, default='none', required=True, help='Template format prompt for extracting notes.')
    args = parser.parse_args()
  main(args.transcript,args.notes_prompt)
