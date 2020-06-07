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
import logging
from functools import partial
from demo_utils import download_model_folder
import argparse
import subprocess as sp

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import boolean_string

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
        logging.info('data parallel because more than one gpu')
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

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument("--fp16", type=boolean_string, default=False)
    
    parser.add_argument("--generation_length", type=int, default=20)
    parser.add_argument("--max_history", type=int, default=2)

    parser.add_argument("--number_of_examples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.0)

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #### load the GPT-2 model 
    config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args.n_gpu, args.device, args.fp16, verbose=True)
    model.to(device)
    model.eval()

    history = []

    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            raw_text = input("USR >>> ")
        history.append(raw_text)

        context_ids = sum([tokenizer.encode(h) + [EOS_ID] for h in history],[])

        samples = sample_sequence(model, tokenizer, context_ids, 
                                  device, args.number_of_examples, args.generation_length, 
                                  args.temperature, args.top_k, args.top_p)
        samples = samples[:, len(context_ids):].tolist()

        texts = []
        for sample in samples:
            text = tokenizer.decode(sample, clean_up_tokenization_spaces=True)
            text = text[: text.find(tokenizer.eos_token)]
            texts.append(text)

        for t in texts:
            print("    >>>", t)
        
        print("BOT >>>", texts[0])
        history.append(texts[0])
        history = history[-(2*args.max_history+1):]


if __name__ == '__main__':

    PYTHON_EXE = 'python'
    MODEL_FOLDER = './models'
    DATA_FOLDER = './data'

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
    )
    logger = logging.getLogger(__name__)


    if os.path.exists(MODEL_FOLDER):
        logger.info('Found existing ./models folder, skip creating a new one!')
        os.makedirs(MODEL_FOLDER, exist_ok=True)
    else:
        os.makedirs(MODEL_FOLDER)

    #########################################################################
    # Download Model
    #########################################################################
    logger.info('Downloading models...')
    download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)

    # model size:  could be one of 'small' (GPT2 with 117M), 'medium'(345M) or 'large' (1542M)
    # dataset: one of 'multiref' or 'dstc'
    # from_scratch: True : load model trained from scratch or False: load model trained from fine-tuning the GPT-2
    target_folder = download_model(model_size='small', dataset='multiref', from_scratch=False)
    logger.info('Done!\n')
    
    run_model()
