import json
from os.path import abspath, dirname, exists, join
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket
import os, sys
import re
import string
from functools import partial
from demo_utils import download_model_folder
import argparse
import subprocess as sp
from nltk.corpus import stopwords

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import boolean_string
from get_embedding import download_embedding

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# make sure it's the token id of <|endoftext|>
EOS_ID = 50256

def load_model(model, checkpoint, n_gpu, device, fp16, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        #start_model = model
        #if (hasattr(model, "transformer")
        #    and all(not s.startswith('transformer.')
        #            for s in model_state_dict.keys())):
        #    logger.info('loading transformer only')
        #    start_model = model.transformer
        model.load_state_dict(model_state_dict)

    if fp16:
        logger.info('in fp16, model.half() activated')
        model.half()
    model.to(device)
    if n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model


def fix_state_dict_namespace(model_state_dict):
    #print("Old keys: ", model_state_dict.keys())
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    model_state_dict['lm_head.weight'] = model_state_dict['lm_head.decoder.weight']
    model_state_dict.pop('lm_head.decoder.weight', None)

    #print("New keys: ", model_state_dict.keys())
    return model_state_dict


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tokenizer, context_ids, device, num_samples, max_length, temperature, top_k, top_p):
    # Parse parameters
    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
    context_tensor = context_tensor.unsqueeze(0).repeat(num_samples, 1)
    generated = context_tensor
    with torch.no_grad():
        while True:
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0.0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if (generated[:, len(context_ids):] == tokenizer.eos_token_id).any(dim=1).all():
                # EOS token id found in each sample
                break
            if generated.shape[1] - len(context_ids) >= max_length:
                # Maximum length reached
                break
    return generated

additional_stopwords = ["okay", "ok", "yeah"]
re_stopwords = re.compile(r'\b(' + r'|'.join(stopwords.words('english') + additional_stopwords) + r')\b\s*')
re_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
def filter_utterances(utterance_list):
    def process_utterance(utt):
        utt = utt.lower()
        utt = re_stopwords.sub('', utt)
        utt = re_punctuation.sub('', utt)
        return utt
        
    processed_utterance_list = [process_utterance(x) for x in utterance_list]

    utterance_list = [x[0] for x in zip(utterance_list, processed_utterance_list) if len(x[1]) > 8]
    return utterance_list

def cluster_utterances(utterance_list):
    grouping_similarity_threshold = 0.7
    def get_similar(sent_vectors, idx, idx_start, threshold):
        # get similar vectors starting from idx_start to specified vector at idx
        assert 0 <= idx < idx_start <= len(sent_vectors), "Invalid values %d %d %d" % (idx, idx_start, len(sent_vectors))
        if idx_start == len(sent_vectors):
            return

        sim_list = np.dot(sent_vectors[idx_start:], sent_vectors[idx])

        for i, s in enumerate(sim_list):
            if s >= threshold:
                yield idx_start + i, s

    utterance_list = filter_utterances(utterance_list)

    sent_vectors = download_embedding(utterance_list)
    assert len(sent_vectors) == len(utterance_list)

    # normalize sentence vectors
    sent_vectors = sent_vectors / np.linalg.norm(sent_vectors, axis=1, keepdims=True)

    # initialize the cluster result as one utterance per cluster
    cluster_result = [{"utterances": [i], "score": 1.0} for i, _ in enumerate(utterance_list)]

    for idx1, result1 in enumerate(cluster_result):
        if "cluster" in result1:
            # this result was already merged to another result
            # skip it
            continue

        cluster_id = idx1

        result1["cluster"] = cluster_id  # as the cluster id
        for idx2, sim in get_similar(sent_vectors, idx1, idx1 + 1, grouping_similarity_threshold):
            assert idx1 < idx2

            result2 = cluster_result[idx2]
            if "cluster" in result2:
                continue

            result2["cluster"] = cluster_id
            result1["utterances"] += result2["utterances"]
            result1["score"] += result2["score"]

    # getting utterances whose cluster id is equal to the utterance id, i.e. the first utterance of the cluster
    cluster_result = [{"utterances": x["utterances"], "score": x["score"]} for x in cluster_result if x["cluster"] == x["utterances"][0]]
    cluster_result = sorted(cluster_result, key=lambda x: x["score"], reverse=True)

    all_text = [utterance_list[x["utterances"][0]] for x in cluster_result]
    popularity = [x["score"] for x in cluster_result]
    return all_text, popularity

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument("--fp16", type=boolean_string, default=False)
    
    parser.add_argument("--generation_length", type=int, default=20)

    parser.add_argument("--number_of_examples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.0)

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--gpu", type=int, default=0)

    return parser


def random_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Interact(object):
    def __init__(self):
        parser = create_arg_parser()
        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
        self.args = args

        random_seed(args.seed)

        #### load the GPT-2 model 
        config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

        model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args.n_gpu, args.device, args.fp16, verbose=True)
        model.to(device)
        model.eval()
        self.model = model

    def get_response(self, history):
        context_ids = sum([self.tokenizer.encode(h) + [EOS_ID] for h in history],[])

        samples = sample_sequence(self.model, self.tokenizer, context_ids, 
                                  self.args.device, self.args.number_of_examples, self.args.generation_length, 
                                  self.args.temperature, self.args.top_k, self.args.top_p)

        samples = samples[:, len(context_ids):].tolist()

        texts = []
        for sample in samples:
            text = self.tokenizer.decode(sample, clean_up_tokenization_spaces=True)
            text = text[: text.find(self.tokenizer.eos_token)]
            texts.append(text)

        clustered_texts, popularities = cluster_utterances(texts)

        return clustered_texts, popularities


def run_model():

    interact_object = Interact()

    history = []

    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            raw_text = input("USR >>> ")

        history.append(raw_text)

        clustered_texts, popularities = interact_object.get_response(history)

        for t, p in zip(clustered_texts, popularities):
            print("    >>>", p, t)
        
        print("BOT >>>", clustered_texts[0])
        history.append(clustered_texts[0])
        history = history[-4:]

if __name__ == '__main__':
    run_model()


