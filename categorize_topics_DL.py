# using https://huggingface.co/facebook/bart-large-mnli
import os
from pprint import pprint as pp
# defaultdic
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

category_labels = ['date', 'geopolitical entity']
philosophical_field = [
    "Aesthetics",
    "Epistemology",
    "Ethics",
    "Logic",
    "Metaphysics",
    "Philosophy of mind",
    "Philosophy of science",
    "Meta-philosophy",
    "Philosophy of education",
    "Philosophy of history",
    "Philosophy of language",
    "Philosophy of law",
    "Philosophy of mathematics",
    "Philosophy of religion",
    "Political philosophy",
    "Environmental philosophy",
    "Feminism"
]

dirs = list(os.walk('transcripts'))
transcript_dirs = [dir for dir in dirs[0][2]]

for transcript_dir in transcript_dirs:
    categories = defaultdict(list)
    fields = defaultdict(list)
    with open(f'transcripts/{transcript_dir}', 'r') as f:
        text = f.read()
    
    text = text.split()
    for word in tqdm(text, desc=f'Classifying words for {transcript_dir}'):
        category_classification = classifier(word, category_labels, multi_label=True)
        for label, score in zip(category_classification['labels'], category_classification['scores']):
            if score > 0.9:
                categories[label].append(word)

    field_classification = classifier(text, philosophical_field, multi_label=True)
    for label, score in zip(category_classification['labels'], category_classification['scores']):
        if score > 0.9:
            fields[label].append(word)

    # save it to classification/transcript_dir
    with open(f'classification/{transcript_dir}_categories.txt', 'w') as f:
        f.write(f'Categories: {categories}')
    
    with open(f'classification/{transcript_dir}_fields.txt', 'w') as f:
        f.write(f'Fields: {fields}')
        

        






