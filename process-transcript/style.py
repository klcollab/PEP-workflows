from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

def main(txtfn,pfn):
  with open(txtfn,'r') as txtf:
    txt = txtf.read()
  with open(pfn,'r') as pf:
    pt = pf.read()
  # ignore 'llama4:17b-scout-16e-instruct-q8_0', 'llama4:16x17b'
  for modelname in ['llama3.3:70b-instruct-q8_0', 'tulu3:70b', 'phi4:latest', 'qwen3:32b-fp16']:
    print('Running using '+modelname)
    model = OllamaLLM(model=modelname, temperature=0.0, num_predict=-1)
    prompt = PromptTemplate.from_template(pt)
    chain_model = prompt | model | StrOutputParser()
    response = chain_model.invoke({'transcript': txt}).strip()
    now = datetime.now()
    with open('style-'+modelname+'-'+str(now.date())+'-'+str(now.hour)+str(now.minute)+str(now.second)+'.txt','w') as rf:
      rf.write(response+'\n')

if __name__ == '__main__':
  import argparse
  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', '--transcript', type=str, default='none', required=False, help='Interview transcript.')
    parser.add_argument('-p', '--prompt', type=str, default='none', required=False, help='Template formatted prompt.')
    args = parser.parse_args()
    if args.transcript == 'none' or args.prompt == 'none':
      parser.parse_args(['--help'])
  txtfn = args.transcript
  promptfn = args.prompt
  main(txtfn,promptfn)
