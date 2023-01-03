# obtain transcripts from: https://www.philosophizethis.org/transcripts
# store in transcripts folder

from bs4 import BeautifulSoup
import requests
import time
from pprint import pprint as pp

URL = 'https://www.philosophizethis.org'

r = requests.get(f'{URL}/transcripts')
  
soup = BeautifulSoup(r.content, 'html5lib')

# find class='archive-item-link'

r = soup.find_all('a', attrs = {'class':'archive-item-link'})
episodes = [link['href'] for link in r]
# extract last part of the url
episodes = [episode.split('/')[-1] for episode in episodes]

for episode_dir in episodes:
    episode_url = f'{URL}/transcript/{episode_dir}'
    print(episode_url)
    r = requests.get(episode_url)
    soup = BeautifulSoup(r.content, 'html5lib')
    # get all the paragraphs in order
    paragraphs = soup.find_all('p')
    # get the text from each paragraph
    text = [p.text for p in paragraphs]
    # remove empty strings
    text = [t for t in text if t]
    # remove last 6 elements, they are not part of the transcript
    text = text[:-6]
    text = '\n'.join(text)

    # store in the transcripts folder as a text file with name episode_url
    with open(f'transcripts/{episode_dir}.txt', 'w') as f:
        f.write(text)
    
    # wait for 1 second to avoid clogging the server
    time.sleep(1)
