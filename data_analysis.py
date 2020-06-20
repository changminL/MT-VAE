import transformers
import os
from collections import OrderedDict
import torch
import importlib
import os, logging, shutil
from tqdm import tqdm
from zipfile import ZipFile
import requests
import json
import pickle

file_path = '/mnt/nas2/newstest/wmt17_de_en/'

src = 'de'
tgt = 'en'
data = 'train'

target_sentences = list()

with open(file_path + data + '.' + tgt, "r", encoding='utf-8') as f:
    for line in f:
        target_sentence = line.rstrip("\r\n")
        target_sentences.append(target_sentence)

f.close()

def http_get(url, path):
    with open(path, "wb") as file_binary:
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
            req.raise_for_status()

        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)
    progress.close()


__DOWNLOAD_SERVER__ = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/'

model_name_or_path = __DOWNLOAD_SERVER__ + 'bert-base-nli-mean-tokens' + '.zip'

if model_name_or_path.startswith('http://') or model_name_or_path.startswith('https://'):
    model_url = model_name_or_path
    folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250]

    try:
        from torch.hub import _get_torch_home
        torch_cache_home = _get_torch_home()
    except ImportError:
        torch_cache_home = os.path.expanduser(
            os.getenv('TORCH_HOME', os.path.join(
                os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
    default_cache_path = os.path.join(torch_cache_home, 'sentence_transformers')
    model_path = os.path.join(default_cache_path, folder_name)
    os.makedirs(model_path, exist_ok=True)

    if not os.listdir(model_path):
        if model_url[-1] == "/":
            model_url = model_url[:-1]
        logging.info("Downloading sentence transformer model from {} and saving it at {}".format(model_url, model_path))
        try:
            zip_save_path = os.path.join(model_path, 'model.zip')
            http_get(model_url, zip_save_path)
            with ZipFile(zip_save_path, 'r') as zip:
                zip.extractall(model_path)
        except Exception as e:
            shutil.rmtree(model_path)
            raise e


if model_path is not None:
    logging.info("Load SentenceTransformer from folder: {}".format(model_path))

    if os.path.exists(os.path.join(model_path, 'config.json')):
        with open(os.path.join(model_path, 'config.json')) as fIn:
            config = json.load(fIn)
            if config['__version__'] > __version__:
                logging.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))

    with open(os.path.join(model_path, 'modules.json')) as fIn:
        contained_modules = json.load(fIn)

tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(model_path, contained_modules[0]['path']))
bert = transformers.BertModel.from_pretrained(os.path.join(model_path, contained_modules[0]['path'])).cuda(0)

embeddings = list()

for target_sentence in tqdm(target_sentences):
    tokenized_input = tokenizer.encode_plus(target_sentence, max_length=512)
    input_ids = torch.tensor(tokenized_input['input_ids'], dtype=torch.long).unsqueeze(0).cuda(0)
    attention_mask = torch.tensor(tokenized_input['attention_mask'], dtype=torch.long).unsqueeze(0).cuda(0)
    token_type_ids = torch.tensor(tokenized_input['token_type_ids'], dtype=torch.long).unsqueeze(0).cuda(0)
    outputs = bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    del input_ids, attention_mask, token_type_ids
    embedding = outputs[0].mean(1).squeeze(0).cpu()
    embeddings.append(embedding.data)

with open('/mnt/nas2/newstest/embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f)