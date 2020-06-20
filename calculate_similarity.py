import pickle
import torch
import logging
from tqdm import tqdm

file_path = '/Volumes/share/newstest/wmt17_de_en/'

logging.info("Sentence dictionary is loading")
with open(file_path+'idx2text.pkl', 'rb') as f:
    target_sentences = pickle.load(f)
logging.info("Sentence dictionary is loaded!")

logging.info("Embeddings are loading")
with open('/mnt/nas2/newstest/embeddings.pickle', 'rb') as f:
    embeddings = pickle.load(f)
logging.info("Embeddings are loaded!")

logging.info("Similiarity is calculating")
num_embeddings = len(embeddings)
total_sim = []
for i in tqdm(range(num_embeddings)):
    i_sim = []
    for j in range(num_embeddings):
        if i == j:
            continue
        else:
            sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
            if sim > 0.91 and sim < 1.0:
                i_sim.append(((i, j), sim))
    total_sim.append(i_sim)
logging.info("Similiarity is calculated!")

import csv

f = open(file_path+'similiarity.csv', 'w', encoding='utf-8', newline="")
wr = csv.writer(f)

for i_sim in total_sim:
    i_length = len(i_sim)
    if i_length <= 5 and i_length > 0:
        for i in range(i_length):
            (sent_idx1, sent_idx2), sim = i_sim[i]

            write_elements = [sent_idx1, target_sentences[sent_idx1], sent_idx2, target_sentences[sent_idx2], sim]
            wr.writerow(write_elements)
    elif i_length > 5:
        i_sim_sorted = i_sim.sort(key=lambda x: x[1], reversed=True)
        i_sim_sorted = i_sim_sorted[:5]
        for i in range(len(i_sim_sorted)):
            (sent_idx1, sent_idx2), sim = i_sim_sorted[i]

            write_elements = [sent_idx1, target_sentences[sent_idx1], sent_idx2, target_sentences[sent_idx2], sim]
            wr.writerow(write_elements)

logging.info("Wrote!")
f.close()