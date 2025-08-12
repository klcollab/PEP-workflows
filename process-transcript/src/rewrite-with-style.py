from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

modelnames = ['llama3.3:70b-instruct-q8_0', 'phi4:14b', 'llama4:16x17b','zephyr:7b','wizardlm2:8x22b']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--style_example', type=str, default='none', required=True, help='Document written in the target style.')
parser.add_argument('-d', '--document', type=str, default='none', required=True, help='Document to be rewritten')
parser.add_argument('-p', '--prompt', type=str, default='none', required=True, help='Prompt for the style rewrite.')
args = parser.parse_args()

with open(args.style_example,'r') as f: style = f.read()
with open(args.document,'r') as f: doc = f.read()
with open(args.notes_prompt,'r') as f: ptxt = f.read()

now = datetime.now()
for model in modelnames:
  model = OllamaLLM(model=model, temperature=0.0, num_predict=-1)
  prompt = PromptTemplate.from_template(ptxt)
  chain_model = prompt | model | StrOutputParser()
  rewrite = chain_model.invoke({'style': style, 'doc': doc}).strip()
  with open('rewrite-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.txt','w') as f: f.write(rewrite)

