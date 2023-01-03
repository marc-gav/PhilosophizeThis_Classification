# using https://huggingface.co/facebook/bart-large-mnli
import os
from pprint import pprint as pp
# defaultdic
from collections import defaultdict
from tqdm import tqdm
import torch
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', max_length=1024)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
nli_model.to(device)

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

def obtain_chunks(text, model_max_len=1024):
    # we want to chunk the text into sentences that are shorter than the max length of the model
    sentences = nltk.sent_tokenize(text)
    sentences_token_length = []
    for sentence in sentences:
        sentences_token_length.append(len(tokenizer.encode(sentence)))
    # now we have a list of the token length of each sentence
    # we want to chunk the text into sentences that are shorter than the max length of the model
    # we can do this by adding the token length of each sentence until we reach the max length
    # then we can start a new chunk
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for sentence, sentence_token_length in zip(sentences, sentences_token_length):
        if current_chunk_length + sentence_token_length > model_max_len:
            chunks.append(current_chunk)
            current_chunk = []
            current_chunk_length = 0
        current_chunk.append(sentence)
        current_chunk_length += sentence_token_length
    chunks.append(current_chunk)
    return chunks

def classify_text(text, labels):
    hypothesis_template = "This text is about {}."
    scores = [0]*len(labels)
    hypotheses = [hypothesis_template.format(label) for label in labels]
    for chunk in tqdm(obtain_chunks(text), desc='Classifying labels'):
        premise = ' '.join(chunk)
        for pos, hypothesis in enumerate(hypotheses):
            x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                truncation_strategy='only_first')
            logits = nli_model(x.to(device))[0]
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:,1].item()

            # we want to do a max over the labels
            current_score = scores[pos]
            max_score = max(current_score, prob_label_is_true)
            scores[pos] = max_score

    return {
        'labels': labels,
        'scores': scores
    }


dirs = list(os.walk('transcripts'))
transcript_dirs = [dir for dir in dirs[0][2]]

for transcript_dir in tqdm(transcript_dirs, desc=f'Classifying transcripts'):
    categories = defaultdict(list)
    fields = []
    with open(f'transcripts/{transcript_dir}', 'r') as f:
        text = f.read()

    # tokenize text for the classifier
    
    field_classification = classify_text(text, philosophical_field)
    print(transcript_dir)
    pp(field_classification)
    input()
    for label, score in zip(field_classification['labels'], field_classification['scores']):
        if score > 0.92:
            fields.append(label)
    
    with open(f'classification/{transcript_dir}_fields.txt', 'w') as f:
        f.write(f'Fields: {fields}')
        

        






