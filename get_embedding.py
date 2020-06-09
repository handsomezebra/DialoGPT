
import os
import os.path as op
from tqdm import tqdm
import numpy as np
import math

from file_util import load_data, save_data, make_sure_path_exists

remote_sentence_bert = os.getenv("REMOTE_SENTENCE_BERT")

if remote_sentence_bert:
    from .remote_api import call_sentence_bert
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    call_sentence_bert = lambda x: model.encode(x, batch_size=64, show_progress_bar=False)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def download_embedding(sentence_list, batch_size=8192):
    embedding_list = []
    for chunk in chunks(sentence_list, batch_size):
        embeddings = call_sentence_bert(chunk)
        embeddings = np.array(embeddings, dtype=np.float16)
        embedding_list.append(embeddings)

    all_embeddings = np.concatenate(embedding_list)
    assert len(sentence_list) == len(all_embeddings)

    return all_embeddings

def get_embedding_for_sentence(sentence):
    emb = download_embedding([sentence])

    return emb[0]


if __name__ == "__main__":
    data = [x[0] for x in load_data("agent_filtered_cluster_result.csv")]

    embeddings = download_embedding(data)
    save_data(embeddings, "agent_filtered_cluster_sentence.npy")
