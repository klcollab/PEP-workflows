##
# Clinical Utility Report
## 
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from pathlib import Path

modelnames = ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:14b', 'zephyr:7b', 'llama4:16x17b', 'gemma3:27b','olmo2:13b','wizardlm2:8x22b','granite4:3b','mistral-small3.2:24b','cogito:70b']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='none', required=True, help='CSV file')
parser.add_argument('-s', '--lpestory', type=str, default='none', required=True, help='LPE Story')
parser.add_argument('-p', '--prompt', type=str, default='none', required=True, help='Prompt')
args = parser.parse_args()

with open(args.data,'r') as f: data = f.read()
with open(args.lpestory,'r') as f: story = f.read()
with open(args.prompt,'r') as f: prompttxt = f.read()

storyid = Path(args.lpestory).stem
now = datetime.now()
for modelname in modelnames:
  model = OllamaLLM(model=modelname, temperature=0.0, num_predict=-1)
  prompt = PromptTemplate.from_template(prompttxt)
  chain_model = prompt | model | StrOutputParser()
  response = chain_model.invoke({'survey_data': data, 'lpe_story': story}).strip()
  with open(storyid+'-cur-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.txt','w') as f: f.write(response)
