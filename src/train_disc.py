import os
import random
import time
import argparse
import yaml
import re
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel  #355M
from transformers import AutoTokenizer, OPTForCausalLM   #350M #arxiv:2205.01068 "facebook/opt-350m"
#from transformers import AutoTokenizer, BloomForCausalLM #560M #arxiv:2211.01786 "bigscience/bloomz-560m"

from transformers import DebertaTokenizer, DebertaModel, DebertaConfig
#from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config

from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from datasets import load_dataset
import evaluate

import utils.datasets
import utils.functions

init_time = time.time()
print(datetime.datetime.today())

#################################################################
# Configuration of DistibutedDataParallel 
#################################################################
accelerator = Accelerator()
device = accelerator.device
print("accelerator.num_processes: " + str(accelerator.num_processes))
#################################################################
# Initialization for reproductions.
#################################################################
init_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--seed",               type=int,   default=10)
parser.add_argument("--dataset_type",       type=str,   default="common_gen") # "common_gen" or "adamlin/roc_story"
parser.add_argument("--model_type",         type=str,   default="gpt2")
parser.add_argument("--method_type",        type=str,   default="disc")
parser.add_argument("--decode_type",        type=str,   default="top-p")
parser.add_argument("--disc_learning_rate", type=float, default=1.0e-7)
parser.add_argument("--gene_model_idx",     type=int,   default=0)
parser.add_argument("--disc_model_idx",     type=int,   default=0)

args               = parser.parse_args()
seed               = args.seed
dataset_type       = args.dataset_type
model_type         = args.model_type
method_type        = args.method_type
decode_type        = args.decode_type
disc_learning_rate = args.disc_learning_rate
gene_model_idx     = args.gene_model_idx
disc_model_idx     = args.disc_model_idx

#################################################################
# Reset random seed
#################################################################
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True # Speedup GPU calculations.
torch.backends.cudnn.deterministic = True

#################################################################
# Load Config
#################################################################
dir_path   = "/home/"
model_path = "/scratch/"
if not os.path.exists(dir_path   + dataset_type + "/data/"   + method_type): os.makedirs(dir_path   + dataset_type + "/data/"   + method_type)
if not os.path.exists(dir_path   + dataset_type + "/figure/" + method_type): os.makedirs(dir_path   + dataset_type + "/figure/" + method_type)
if not os.path.exists(model_path + dataset_type + "/model/"  + method_type): os.makedirs(model_path + dataset_type + "/model/"  + method_type)

with open(dir_path + "config/" + dataset_type + "_" + str(model_type) + ".yaml", "r") as f: config = yaml.load(f, Loader=yaml.Loader)[method_type]
epoch              = config["epoch"]
batch_size         = config["batch_size"]
buffer_num         = config["buffer_num"]
temperature        = config["temperature"]
top_p              = config["top_p"]
max_length         = config["max_length"]
layer_norm_epsilon = config["layer_norm_epsilon"]
disc_alpha         = config["disc_alpha"]
weight_decay       = config["weight_decay"]
max_gradient_norm  = config["max_gradient_norm"]

disc_batch_size = int(batch_size/2)
disc_step_num   = int(buffer_num/disc_batch_size)
gene_step_num   = int(buffer_num/batch_size)

print("==================================================================================")
print("seed        : " + str(seed))
print("dataset_type: " + str(dataset_type))
print("model_type  : " + model_type)
print("method_type : " + method_type)
print("decode_type : " + decode_type)
print("==================================================================================")
print("epoch       : " + str(epoch))
print("batch_size  : " + str(batch_size))
print("top_p       : " + str(top_p))
print("disc_learning_rate: " + str(disc_learning_rate))
print("weight_decay: " + str(weight_decay))
print("==================================================================================")

#################################################################
# Configures of Tokenizer and Dataset.
#################################################################
# Tokenizer and Dataset Parameters
temp_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
disc_tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base", #"microsoft/deberta-v2-xlarge"
                                                  add_special_tokens=False,
                                                  bos_token=temp_tokenizer.bos_token,
                                                  eos_token=temp_tokenizer.eos_token)
if model_type=="gpt2"   : gene_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium",
                                                                         add_special_tokens=False,
                                                                         sep_token=disc_tokenizer.sep_token,
                                                                         pad_token=disc_tokenizer.pad_token,
                                                                         bos_token=temp_tokenizer.bos_token,
                                                                         eos_token=temp_tokenizer.eos_token)
elif model_type=="opt"  : gene_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m",
                                                                         add_special_tokens=False,
                                                                         sep_token=disc_tokenizer.sep_token,
                                                                         pad_token=disc_tokenizer.pad_token,
                                                                         bos_token=temp_tokenizer.bos_token,
                                                                         eos_token=temp_tokenizer.eos_token)
elif model_type=="bloom": gene_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m",
                                                                         add_special_tokens=False,
                                                                         sep_token=disc_tokenizer.sep_token,
                                                                         pad_token=disc_tokenizer.pad_token,
                                                                         bos_token=temp_tokenizer.bos_token,
                                                                         eos_token=temp_tokenizer.eos_token)
print("==================================================================================")

#################################################################
# Dataset / Reset random seed
# "common_gen" or "adamlin/roc_story"
#################################################################
if dataset_type=="common_gen" : dataset = load_dataset(dataset_type)
elif dataset_type=="roc_story": dataset = load_dataset("csv", data_files={"train"     : dir_path + "dataset/roc_train_10seed.csv",
                                                                          "validation": dir_path + "dataset/roc_valid_10seed.csv"})
#################################################################
# Reset random seed
#################################################################
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True # Speedup GPU calculations.
torch.backends.cudnn.deterministic = True
#################################################################
train_dataset = utils.datasets.Seq2SeqDataset(dataset=dataset["train"],
                                              gene_tokenizer=gene_tokenizer,
                                              disc_tokenizer=disc_tokenizer,
                                              dataset_type=dataset_type)

#################################################################
# Configures of Finetuning.
#################################################################
data_num            = len(train_dataset)
replay_num          = int(epoch*data_num/buffer_num)
disc_training_steps = replay_num*disc_step_num
disc_warmup_steps   = int(0.2*replay_num*disc_step_num)

#################################################################
# Instance Generator and Optimizer
#################################################################
if model_type=="gpt2"   : gene_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
elif model_type=="opt"  : gene_model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
elif model_type=="bloom": gene_model = BloomForCausalLM.from_pretrained("bigscience/bloomz-560m")

gene_model.resize_token_embeddings(len(gene_tokenizer))
gene_model.layer_norm_epsilon = layer_norm_epsilon
gene_model.temperature = temperature
for p in gene_model.parameters(): p.grad = None

class DebertaClassifier(nn.Module):
    def __init__(self):
        super(DebertaClassifier, self).__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.linear = nn.Linear(self.deberta.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask=None):
        output = self.deberta(input_ids=input_ids,
                              #position_ids=attention_mask.cumsum(axis=-1)-1,
                              attention_mask=attention_mask)
        output = output.last_hidden_state[:, 0, :]
        output = self.linear(output)
        return output
disc_model = DebertaClassifier()
for p in disc_model.parameters(): p.grad = None
disc_optimizer = torch.optim.AdamW(disc_model.parameters(),
                                   lr=disc_learning_rate,
                                   weight_decay=weight_decay,
                                   maximize=True)
#################################################################
gene_model.train()
disc_model.train()
#################################################################
print("gene_model_size: " + str(int(gene_model.num_parameters()*1e-6))         + "M parameters")
print("disc_model_size: " + str(int(disc_model.deberta.num_parameters()*1e-6)) + "M parameters")
print("==================================================================================")

#################################################################
# Scheduler and Loss Fucntion
#################################################################
disc_scheduler = get_linear_schedule_with_warmup(disc_optimizer,
                                                 num_warmup_steps=disc_warmup_steps,
                                                 num_training_steps=disc_training_steps)
gene_model, disc_model, disc_optimizer = accelerator.prepare(gene_model, disc_model, disc_optimizer)

#################################################################
# Train
#################################################################
if gene_model_idx>=0: gene_model.load_state_dict(torch.load(model_path + dataset_type + "/model/sft/gene_model_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx.pth"))
else                : gene_model.load_state_dict(torch.load(model_path + dataset_type + "/model/sft/gene_model_" + str(model_type) + "_" + str(seed) + "seed.pth"))
for p in gene_model.parameters(): p.grad = None
rand_idx = torch.load(dir_path + dataset_type + "/data/rand_idx_" + str(seed) +  "seed.pth")
sample_rand_idx = rand_idx["rand_idx_2"][:(epoch+1)*data_num]

#################################################################
# Loss Log
ce_criterion = nn.CrossEntropyLoss(ignore_index=gene_tokenizer.pad_token_id, reduction="mean")
softmax      = nn.Softmax(dim=-1)
log_softmax  = nn.LogSoftmax(dim=-1)

train_ce_loss_log = []
valid_ce_loss_log = []
sft_ce_loss_log   = []

bleu_score = evaluate.load("bleu")
gene_bleu_score_log     = []
gene_distinct_score_log = []
sft_bleu_score_log      = []
sft_distinct_score_log  = []
#bert_score = evaluate.load("bertscore")
#bert_score_log = []

disc_loss_log          = []
confid_gene_mean_log   = []
confid_gene_median_log = []
confid_gene_upperQ_log = []
confid_gene_lowerQ_log = []
confid_gene_max_log    = []
confid_gene_min_log    = []

print("####################################################################################################################################################")
for replay_itr in range(replay_num):
    ##################################################################################################################################
    # Sample Buffer
    ##################################################################################################################################
    ppo_real_data_buffer = []

    ppo_real_text_buffer = []
    ppo_gene_text_buffer = []
    disc_real_text_buffer = []
    disc_gene_text_buffer = []
    
    ppo_real_confid_buffer = []
    ppo_confid_gene_buffer = []

    for step_itr in range(buffer_num): ppo_real_data_buffer.append(train_dataset[sample_rand_idx[step_itr+replay_itr*buffer_num]])
    ##################################################################################################################################
    # Plot
    ##################################################################################################################################
    if replay_itr%20==0:
        print("{}/{} replay_num: {:.2f} mins.".format(replay_itr, replay_num, (time.time()-init_time)/60))
        utils.functions.plot_result(replay_itr=replay_itr,
                                    gene_step_num=gene_step_num,
                                    seed=seed,
                                    dir_path=dir_path,
                                    dataset_type=dataset_type,
                                    model_type=model_type,
                                    method_type=method_type,
                                    confid_gene_upperQ_log=confid_gene_upperQ_log,
                                    confid_gene_lowerQ_log=confid_gene_lowerQ_log,
                                    confid_gene_max_log=confid_gene_max_log,
                                    confid_gene_min_log=confid_gene_min_log,
                                    confid_gene_median_log=confid_gene_median_log,
                                    gene_model_idx=gene_model_idx,
                                    disc_model_idx=disc_model_idx)
    
    ##################################################################################################################################
    # Buffer Generated, Real Text, Confidence, and Reward
    ##################################################################################################################################
    # Buffer Generated and Real Text
    # str -> gene_tokenizer -> id| id-> gene_generator -> id
    #################################################################
    for step_itr in range(gene_step_num):
        #################################################################
        # Decode Sentence
        #################################################################
        ppo_gene_batch = train_dataset.collate_fn(ppo_real_data_buffer[step_itr*batch_size:(step_itr+1)*batch_size])
        if model_type=="gpt2": ppo_gene_source_input_ids = torch.cat((ppo_gene_batch["g_source_input_ids_padded"], \
                                                                      gene_tokenizer.sep_token_id * torch.ones((batch_size, 1), dtype=torch.long)),
                                                                      dim=1)
        else                 : ppo_gene_source_input_ids = torch.cat((ppo_gene_batch["g_source_input_ids_padded"], \
                                                                      gene_tokenizer(" ", add_special_tokens=False)["input_ids"][0] * torch.ones((batch_size, 1), dtype=torch.long), \
                                                                      gene_tokenizer.sep_token_id * torch.ones((batch_size, 1), dtype=torch.long)),
                                                                      dim=1)
        ppo_gene_source_mask = torch.where(ppo_gene_source_input_ids==gene_tokenizer.pad_token_id, 0, 1)
        ppo_real_batch = train_dataset.collate_fn(ppo_real_data_buffer[step_itr*batch_size:(step_itr+1)*batch_size])        
        with torch.no_grad():
            if decode_type=="top-p" : ppo_gene_joint_input_ids = gene_model.generate(ppo_gene_source_input_ids.to(device),
                                                                                     max_length=max_length,
                                                                                     do_sample=True,
                                                                                     early_stopping=True,
                                                                                     temperature=temperature,
                                                                                     top_p=top_p,
                                                                                     #repetition_penalty=1.0,
                                                                                     pad_token_id=gene_tokenizer.pad_token_id,
                                                                                     bos_token_id=gene_tokenizer.bos_token_id,
                                                                                     eos_token_id=gene_tokenizer.eos_token_id,
                                                                                     #length_penalty=1.0,
                                                                                     attention_mask=ppo_gene_source_mask.to(device),
                                                                                     remove_invalid_values=True)
            elif decode_type=="beam": ppo_gene_joint_input_ids = gene_model.generate(ppo_gene_source_input_ids.to(device),
                                                                                     max_length=max_length,
                                                                                     num_beams=4,
                                                                                     do_sample=True,
                                                                                     early_stopping=True,
                                                                                     temperature=temperature,
                                                                                     #repetition_penalty=1.0,
                                                                                     pad_token_id=gene_tokenizer.pad_token_id,
                                                                                     bos_token_id=gene_tokenizer.bos_token_id,
                                                                                     eos_token_id=gene_tokenizer.eos_token_id,
                                                                                     #length_penalty=1.0,
                                                                                     attention_mask=ppo_gene_source_mask.to(device),
                                                                                     remove_invalid_values=True)
        ppo_real_joint_text_ = gene_tokenizer.batch_decode(ppo_real_batch["g_joint_input_ids_padded"], add_special_tokens=False)
        ppo_gene_joint_text_ = gene_tokenizer.batch_decode(ppo_gene_joint_input_ids,                   add_special_tokens=False)
        del ppo_real_batch, ppo_gene_source_input_ids, ppo_gene_joint_input_ids, ppo_gene_source_mask
        torch.cuda.empty_cache()

        ppo_real_joint_text = []
        ppo_gene_joint_text = []
        for item in ppo_real_joint_text_:
            item = item.replace(gene_tokenizer.pad_token, "")
            item = re.split(" +", item)
            item = " ".join(item)
            item = item.strip()
            ppo_real_joint_text.append(item)
        for item in ppo_gene_joint_text_:
            item = item.replace(gene_tokenizer.pad_token, "")
            item = re.split(" +", item)
            item = " ".join(item)
            item = item.strip()
            ppo_gene_joint_text.append(item)
        ppo_real_text_buffer.extend(ppo_real_joint_text)
        ppo_gene_text_buffer.extend(ppo_gene_joint_text)
        del ppo_real_joint_text_, ppo_gene_joint_text_, ppo_real_joint_text,  ppo_gene_joint_text
        torch.cuda.empty_cache()
    ##################################################################################################################################
    # GAN Training
    ##################################################################################################################################
    if replay_itr%20==0:
        print("real: " + ppo_real_text_buffer[0])
        print("ppo : " + ppo_gene_text_buffer[0])
        print("####################################################################################################################################################")
    ##################################################################################################################################
    # Discriminator Training
    ##################################################################################################################################    
    disc_gene_text_buffer, disc_real_text_buffer = utils.functions.shuffle_pair(gene_text_buffer=ppo_gene_text_buffer,
                                                                                real_text_buffer=ppo_real_text_buffer)
    for step_itr in range(disc_step_num):
        disc_real_batch = disc_real_text_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size]
        disc_gene_batch = disc_gene_text_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size]
        #################################################################
        # Discriminator Confidence
        #################################################################
        disc_joint_text_ = []
        disc_joint_text_.extend(disc_real_batch)
        disc_joint_text_.extend(disc_gene_batch)
        del disc_real_batch, disc_gene_batch
        torch.cuda.empty_cache()
        
        disc_joint_text = []
        for item in disc_joint_text_:
            item = item.replace(gene_tokenizer.sep_token, disc_tokenizer.sep_token, 1)
            disc_joint_text.append(item)
        del disc_joint_text_
        torch.cuda.empty_cache()
        #################################################################
        #################################################################
        disc_mix_source_batch = []
        disc_mix_target_batch = []
        for item in disc_joint_text:
            item    = item.split(gene_tokenizer.sep_token, maxsplit=1)
            item[0] = item[0].strip()
            item[1] = " " + gene_tokenizer.sep_token + item[1]
            disc_mix_source_batch.append(item[0])
            disc_mix_target_batch.append(item[1])
        disc_mix_source_input_ids = gene_tokenizer(disc_mix_source_batch,
                                                   padding=True,
                                                   return_tensors="pt",
                                                   add_special_tokens=False)["input_ids"]
        disc_mix_target_input_ids = gene_tokenizer(disc_mix_target_batch,
                                                   padding=True,
                                                   return_tensors="pt",
                                                   add_special_tokens=False)["input_ids"]
        disc_mix_joint_input_ids  = torch.cat((disc_mix_source_input_ids, disc_mix_target_input_ids), dim=1)
        disc_mix_joint_mask  = torch.where(disc_mix_joint_input_ids==disc_tokenizer.pad_token_id, 0, 1)
        disc_joint_input_ids = disc_mix_joint_input_ids
        if replay_itr%20==0 and step_itr==0:
            print("real PAD: " + gene_tokenizer.decode(disc_joint_input_ids[0]))
            print("ppo  PAD: " + gene_tokenizer.decode(disc_joint_input_ids[-1]))
            print("####################################################################################################################################################")
        #################################################################
        #################################################################
        disc_joint_mask = torch.where(disc_joint_input_ids==disc_tokenizer.pad_token_id, 0, 1)
        del disc_joint_text
        torch.cuda.empty_cache()

        disc_logit = disc_model(disc_joint_input_ids.to(device),
                                attention_mask=disc_joint_mask.to(device))
        real_logit = disc_logit[:disc_batch_size]
        gene_logit = disc_logit[disc_batch_size:]
        #################################################################
        real_log_probs = log_softmax(torch.cat((real_logit[:, 0].unsqueeze(-1), gene_logit[:, 0].unsqueeze(-1)), dim=1))[:, 0]
        gene_log_probs = log_softmax(torch.cat((real_logit[:, 1].unsqueeze(-1), gene_logit[:, 1].unsqueeze(-1)), dim=1))[:, 1]
        del disc_joint_input_ids, disc_joint_mask, disc_logit, real_logit, gene_logit
        torch.cuda.empty_cache()
        #################################################################
        confid_gene_mean_log.append(np.mean(gene_log_probs.exp().tolist()))
        confid_gene_median_log.append(np.median(gene_log_probs.exp().tolist()))
        confid_gene_upperQ_log.append(np.percentile(gene_log_probs.exp().tolist(), q=75))
        confid_gene_lowerQ_log.append(np.percentile(gene_log_probs.exp().tolist(), q=25))
        confid_gene_max_log.append(np.max(gene_log_probs.exp().tolist()))
        confid_gene_min_log.append(np.min(gene_log_probs.exp().tolist()))
        
        #################################################################
        # Discriminator Loss
        #################################################################
        disc_loss = disc_alpha * (real_log_probs[:].sum() + gene_log_probs[:].sum()) / batch_size
        disc_loss_log.append(disc_loss.item())
        del real_log_probs, gene_log_probs
        torch.cuda.empty_cache()
        
        disc_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(disc_model.parameters(), max_gradient_norm)
        disc_optimizer.step()
        disc_scheduler.step()
        disc_optimizer.zero_grad()
        del disc_loss
        torch.cuda.empty_cache()

##################################################################################################################################
# Plot
##################################################################################################################################
print("{}/{} replay_num: {:.2f} mins.".format(replay_itr, replay_num, (time.time()-init_time)/60))
utils.functions.plot_result(replay_itr=replay_itr,
                            gene_step_num=gene_step_num,
                            seed=seed,
                            dir_path=dir_path,
                            dataset_type=dataset_type,
                            model_type=model_type,
                            method_type=method_type,
                            confid_gene_upperQ_log=confid_gene_upperQ_log,
                            confid_gene_lowerQ_log=confid_gene_lowerQ_log,
                            confid_gene_max_log=confid_gene_max_log,
                            confid_gene_min_log=confid_gene_min_log,
                            confid_gene_median_log=confid_gene_median_log,
                            gene_model_idx=gene_model_idx,
                            disc_model_idx=disc_model_idx)

#################################################################
# Save Model
#################################################################
if gene_model_idx>=0:
    torch.save(disc_model.state_dict(),
               model_path + dataset_type + "/model/" + method_type + "/disc_model_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.pth")
    torch.save({"disc_loss_log": disc_loss_log},
               dir_path   + dataset_type + "/data/"  + method_type + "/loss_"       + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.pth")
else:
    torch.save(disc_model.state_dict(),
               model_path + dataset_type + "/model/" + method_type + "/disc_model_" + str(model_type) + "_" + str(seed) + "seed.pth")
    torch.save({"disc_loss_log": disc_loss_log},
               dir_path   + dataset_type + "/data/"  + method_type + "/loss_"       + str(model_type) + "_" + str(seed) + "seed.pth")

print("####################################################################################################################################################")
print("ALL WORK Finished: {:.2f} mins.".format((time.time()-init_time)/60))
print("####################################################################################################################################################")
