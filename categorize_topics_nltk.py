# using https://huggingface.co/facebook/bart-large-mnli
import os
from pprint import pprint as pp
from transformers import pipeline
import nltk
from nltk import word_tokenize
from nltk.chunk import ne_chunk

dirs = list(os.walk('transcripts'))
transcript_dirs = [dir for dir in dirs[0][2]]

classifications = {transcript_dir: {} for transcript_dir in transcript_dirs}

for transcript_dir in transcript_dirs:
    with open(f'transcripts/{transcript_dir}', 'r') as f:
        text = f.read()
    
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    people_names = []
    dates = []
    countries = []
    # Extract the named entities and print the people names
    entities = ne_chunk(tags)
    for entity in entities:
        if hasattr(entity, "label"):
            if entity.label() == "PERSON":
                people_names.append(' '.join([word for word, tag in entity.leaves()]))
            elif entity.label() == "DATE" or entity.label() == "TIME":
                dates.append(' '.join([word for word, tag in entity.leaves()]))
            elif entity.label() == "GPE" or entity.label() == "LOCATION":
                countries.append(' '.join([word for word, tag in entity.leaves()]))
    
    # remove duplicates
    people_names = list(set(people_names))
    dates = list(set(dates))
    countries = list(set(countries))
    classifications[transcript_dir]['authors'] = people_names
    classifications[transcript_dir]['dates'] = dates
    classifications[transcript_dir]['countries'] = countries
    pass

