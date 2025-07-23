from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-story', '--story', type=str, default='none', required=True, help='Prior version of the story.')
parser.add_argument('-n', '--delta_notes', type=str, default='none', required=True, help='Notes taken about the prior version.')
parser.add_argument('-sp', '--story_prompt', type=str, default='none', required=True, help='Template format prompt for revising the story.')
args = parser.parse_args()

with open(args.story,'r') as f: story = f.read()
with open(args.delta_notes,'r') as f: dnotes = f.read()
with open(args.story_prompt,'r') as f: sp = f.read()

modelname = 'llama3.3:70b-instruct-q8_0'
#modelname ='tulu3:70b'
#modelname ='phi4:14b' 
#modelname ='llama4:16x17b'
model = OllamaLLM(model=modelname, temperature=0.0, num_predict=-1)

prompt = PromptTemplate.from_template(sp)
chain_model = prompt | model | StrOutputParser()
print('Refining the story ...')
response = chain_model.invoke({'script': story, 'transcript_notes': dnotes}).strip()
lpestory = response+'\n'
now = datetime.now()
with open('lpestory-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.txt','w') as f: f.write(lpestory)
