import pickle
import torch
import logging
from tqdm import tqdm
import random
from multiprocessing import Pool, Array, Process, Value, Manager
def cal_sim(tid):
    file_path = '/mnt/nas2/newstest/wmt17_de_en/'

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
    cand = random.sample(range(num_embeddings), 100)
    for i in tqdm(cand):
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

    with open('/mnt/nas2/newstest/sim.pkl', 'wb') as f:
        pickle.dump(total_sim, f)

    import csv

    f = open(file_path+'similiarity__'+str(tid)+'.csv', 'w', encoding='utf-8', newline="")
    wr = csv.writer(f)

    for i_sim in total_sim:
        i_length = len(i_sim)
        if i_length <= 10 and i_length > 0:
            for i in range(i_length):
                (sent_idx1, sent_idx2), sim = i_sim[i]

                write_elements = [sent_idx1, target_sentences[sent_idx1], sent_idx2, target_sentences[sent_idx2], sim]
                wr.writerow(write_elements)
        elif i_length > 10:
            i_sim_sorted = sorted(i_sim, key=lambda x: x[1], reverse=True)
            i_sim_sorted = i_sim_sorted[:10]
            for i in range(len(i_sim_sorted)):
                (sent_idx1, sent_idx2), sim = i_sim_sorted[i]

                write_elements = [sent_idx1, target_sentences[sent_idx1], sent_idx2, target_sentences[sent_idx2], sim]
                wr.writerow(write_elements)

    logging.info("Wrote!")
    f.close()

num_threads = 20
t_id = 0
jobs = []
for i in range(num_threads):
    p = Process(target=cal_sim, args=[t_id])
    t_id += 1
    jobs.append(p)

for j in jobs:
    j.start()

for j in jobs:
    j.join()