import pandas as pd
import csv
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--variables', type=str, required=True, help='List of variables to lookup in the data.')
parser.add_argument('records', nargs='+', help='CSV files 1 record per file.')
args = parser.parse_args()

vdf = pd.read_csv(args.variables, index_col=0)

for f in args.records:
  d = pd.read_csv(f,index_col=0)
  cb = d.loc[vdf.index]
  #cb.index = cb.index+'/'+vdf.index
  p = Path(f)
  cb.to_csv(p.stem+'-values.csv',quoting=csv.QUOTE_ALL)
