import pickle
import pandas as pd
import requests
import time
import re

with open('./data/link_df','rb') as f:
    df = pickle.load(f)

for i,link in enumerate(df['link']):
    time.sleep(1)
    audio = requests.get(link)
    filename = link[len(re.sub(r'/[^/]*\.mp3','',link))+1:]
    with open('./audio_files/' + filename, 'wb+') as f:
        f.write(audio.content)
        print('{0}/{1}'.format(i+1,len(df['link'])),filename)
