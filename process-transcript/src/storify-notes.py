import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import pandas as pd

def main(notesfn, storypromptfn, stylefn):
  notesdf = pd.read_csv(notesfn)
  with open(storypromptfn,'r') as ptf: storypt = ptf.read()
  with open(stylefn,'r') as stf: style = stf.read()
  modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:14b', 'qwen3:32b-fp16', 'llama4:16x17b', 'wizardlm2:8x22b', 'olmo2:13b']
  for modelname in modelnames:
    print('Running using '+modelname)
    model = OllamaLLM(model=modelname, temperature=0.0, num_predict=-1)
    story_prompt = PromptTemplate.from_template(storypt)
    story_chain_model = story_prompt | model | StrOutputParser()
    now = datetime.now()
    story = story_chain_model.invoke({'notes': '\n'.join(notesdf['Note']), 'linguistic_style': style}).strip()
    with open('lpestory-'+'-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.txt','w') as rf:
      rf.write(story+'\n')

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--notes', type=str, default='none', required=True, help='CSV file with a note per row.')
    parser.add_argument('-sp', '--story_prompt', type=str, default='none', required=True, help='Template format prompt for telling the story.')
    parser.add_argument('-st', '--linguistic_style', type=str, default='none', required=True, help='Interview transcript.')
    args = parser.parse_args()
  main(args.notes,args.story_prompt,args.linguistic_style)
