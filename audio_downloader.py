import pickle
import pandas as pd
import requests
import time
import re
import sys

n = int(sys.argv[1])

with open('./data.csv','r') as f:
    df = pd.read_csv(f)

for i,link in enumerate(df['link'][n:]):
    time.sleep(1)
    try:
        audio = requests.get(link)
        filename = link[len(re.sub(r'/[^/]*\.(mp3|wav|wma)','',link))+1:]
        print(filename,end=' ')
    except:
        with open('./link_log.txt','+a') as f:
            f.write('Error accessing {}th link\n'.format(n+i))
    try:
        with open('./audio_files/' + filename, '+wb') as f:
            f.write(audio.content)
            print('{0}/{1}'.format(i+1,len(df['link'][n:])))
    except:
        with open('./dl_log.txt','+a') as f:
            f.write('Error reading/saving {}\n'.format(link))
