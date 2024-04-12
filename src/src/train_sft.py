import os
import random
import time
import argparse
import yaml
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import GPT2Tokenizer,  GPT2LMHeadModel        #355M
from transformers import AutoTokenizer,  OPTForCausalLM         #350M #arxiv:2205.01068 "facebook/opt-350m"
#from transformers import AutoTokenizer,  BloomForCausalLM      #560M #arxiv:2211.01786 "bigscience/bloomz-560m"
#from transformers import AutoTokenizer,  AutoModelForSeq2SeqLM #580M #arxiv:2211.01786 "bigscience/mt0-base"
#from transformers import LlamaTokenizer, LlamaForCausalLM

#86M #arxiv:2111.09543 #microsoft/deberta-v3-base
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig
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
# Argparse
#################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--seed",               type=int,   default=10)
parser.add_argument("--dataset_type",       type=str,   default="common_gen") # "common_gen" or "adamlin/roc_story"
parser.add_argument("--model_type",         type=str,   default="gpt2")
parser.add_argument("--method_type",        type=str,   default="sft")
parser.add_argument("--gene_model_idx",     type=int,   default=0)
parser.add_argument("--gene_learning_rate", type=float, default=1.0e-6)
args               = parser.parse_args()
seed               = args.seed
dataset_type       = args.dataset_type
model_type         = args.model_type
method_type        = args.method_type
gene_model_idx     = args.gene_model_idx
gene_learning_rate = args.gene_learning_rate

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
if not os.path.exists(dir_path   + dataset_type)                           : os.makedirs(dir_path   + dataset_type)
if not os.path.exists(dir_path   + dataset_type + "/data/")                : os.makedirs(dir_path   + dataset_type + "/data/")
if not os.path.exists(dir_path   + dataset_type + "/data/"   + method_type): os.makedirs(dir_path   + dataset_type + "/data/"   + method_type)
if not os.path.exists(dir_path   + dataset_type + "/figure/")              : os.makedirs(dir_path   + dataset_type + "/figure/")
if not os.path.exists(dir_path   + dataset_type + "/figure/" + method_type): os.makedirs(dir_path   + dataset_type + "/figure/" + method_type)
if not os.path.exists(model_path + dataset_type)                           : os.makedirs(model_path + dataset_type)
if not os.path.exists(model_path + dataset_type + "/model/")               : os.makedirs(model_path + dataset_type + "/model/")
if not os.path.exists(model_path + dataset_type + "/model/"  + method_type): os.makedirs(model_path + dataset_type + "/model/"  + method_type)
#################################################################
with open(dir_path + "config/" + dataset_type + "_" + str(model_type) + ".yaml", "r") as f: config = yaml.load(f, Loader=yaml.Loader)[method_type]
src_len_flag       = config["src_len_flag"]
epoch              = config["epoch"]
batch_size         = config["batch_size"]
buffer_num         = batch_size
temperature        = config["temperature"]
max_length         = config["max_length"]
layer_norm_epsilon = config["layer_norm_epsilon"]
weight_decay       = config["weight_decay"]
max_gradient_norm  = config["max_gradient_norm"]

print("==================================================================================")
print("seed              : " + str(seed))
print("dataset_type      : " + str(dataset_type))
print("model_type        : " + model_type)
print("method_type       : " + method_type)
print("gene_learning_rate: " + str(gene_learning_rate))
print("gene_model_idx    : " + str(gene_model_idx))
print("==================================================================================")
print("epoch       : " + str(epoch))
print("batch_size  : " + str(batch_size))
print("weight_decay: " + str(weight_decay))
print("==================================================================================")

#################################################################
# Configures of Tokenizer and Dataset.
#################################################################
# Tokenizer and Dataset Parameters
temp_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
disc_tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base",
                                                  add_special_tokens=False,
                                                  bos_token=temp_tokenizer.bos_token,
                                                  eos_token=temp_tokenizer.eos_token)
if model_type=="gpt2": gene_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium",
                                                                      add_special_tokens=False,
                                                                      sep_token=disc_tokenizer.sep_token,
                                                                      pad_token=disc_tokenizer.pad_token,
                                                                      bos_token=temp_tokenizer.bos_token,
                                                                      eos_token=temp_tokenizer.eos_token)
elif model_type=="opt": gene_tokenizer =  AutoTokenizer.from_pretrained("facebook/opt-350m",
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
elif model_type=="mt0": gene_tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-base",
                                                                       add_special_tokens=False,
                                                                       sep_token=disc_tokenizer.sep_token,
                                                                       pad_token=disc_tokenizer.pad_token,
                                                                       bos_token=temp_tokenizer.bos_token,
                                                                       eos_token=temp_tokenizer.eos_token)
elif model_type=="llama": gene_tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_350bt_preview",
                                                                          add_special_tokens=False,
                                                                          sep_token=disc_tokenizer.sep_token,
                                                                          pad_token=disc_tokenizer.pad_token,
                                                                          bos_token=temp_tokenizer.bos_token,
                                                                          eos_token=temp_tokenizer.eos_token)

#################################################################
# Dataset / Reset random seed
# "common_gen"
# "adamlin/roc_story"
#################################################################
if dataset_type=="common_gen" : dataset = load_dataset(dataset_type)
elif dataset_type=="roc_story": dataset = load_dataset("csv", data_files={"train"     : dir_path + "dataset/roc_train_10seed.csv",
                                                                          "validation": dir_path + "dataset/roc_valid_10seed.csv"})
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
valid_dataset = utils.datasets.Seq2SeqDataset(dataset=dataset["validation"],
                                              gene_tokenizer=gene_tokenizer,
                                              disc_tokenizer=disc_tokenizer,
                                              dataset_type=dataset_type)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn)

#################################################################
# Configures of Finetuning.
#################################################################
data_num            = len(train_dataset)
replay_num          = int(epoch*data_num/buffer_num)
gene_step_num       = int(buffer_num/batch_size)
gene_training_steps = replay_num*gene_step_num
gene_warmup_steps   = int(0.2*replay_num*gene_step_num)
print("gene_training_steps: " + str(gene_training_steps))
print("gene_warmup_steps  : " + str(gene_warmup_steps))
print("==================================================================================")

#################################################################
# Instance Generator and Optimizer
#################################################################
if model_type=="gpt2"   : gene_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
elif model_type=="opt"  : gene_model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
elif model_type=="bloom": gene_model = BloomForCausalLM.from_pretrained("bigscience/bloomz-560m")
elif model_type=="mt0"  : gene_model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-base")
elif model_type=="llama": gene_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_350bt_preview")

if model_type=="bart" : gene_model.config.is_encoder_decoder = False
gene_model.resize_token_embeddings(len(gene_tokenizer))
gene_model.layer_norm_epsilon = layer_norm_epsilon
gene_model.temperature = temperature
for p in gene_model.parameters(): p.grad = None
gene_model.train()

if method_type=="sft2": gene_model.load_state_dict(torch.load(model_path + dataset_type + "/model/sft/gene_model_"  + str(model_type) + "_" + str(seed) + "seed.pth"))
gene_optimizer = torch.optim.AdamW(gene_model.parameters(),
                                   lr=gene_learning_rate,
                                   weight_decay=weight_decay,
                                   maximize=False)
print("model_size: " + str(int(gene_model.num_parameters()*1e-6)) + "M parameters")
print("==================================================================================")

#################################################################
# Scheduler and Loss Fucntion
#################################################################
gene_scheduler = get_linear_schedule_with_warmup(gene_optimizer,
                                                 num_warmup_steps=gene_warmup_steps,
                                                 num_training_steps=gene_training_steps)
gene_model, gene_optimizer = accelerator.prepare(gene_model, gene_optimizer)

#################################################################
# Train
#################################################################
if model_type=="gpt2" and method_type=="sft" and gene_model_idx==0:
    rand_idx_0 = []
    rand_idx_1 = []
    rand_idx_2 = []
    rand_idx_3 = []
    rand_idx_4 = []
    rand_idx_5 = []
    rand_idx_6 = []
    rand_idx_7 = []
    rand_idx_8 = []
    rand_idx_9 = []
    for epoch_itr in range(11):
        rand_idx_0.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_1.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_2.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_3.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_4.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_5.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_6.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_7.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_8.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
        rand_idx_9.extend(random.sample(range(len(train_dataset)), k=len(train_dataset)))
    torch.save({"rand_idx_0": rand_idx_0,
                "rand_idx_1": rand_idx_1,
                "rand_idx_2": rand_idx_2,
                "rand_idx_3": rand_idx_3,
                "rand_idx_4": rand_idx_4,
                "rand_idx_5": rand_idx_5,
                "rand_idx_6": rand_idx_6,
                "rand_idx_7": rand_idx_7,
                "rand_idx_8": rand_idx_8,
                "rand_idx_9": rand_idx_9},
                dir_path + dataset_type + "/data/rand_idx_" + str(seed) + "seed.pth")
#################################################################
rand_idx       = torch.load(dir_path + dataset_type + "/data/rand_idx_" + str(seed) + "seed.pth")
ce_rand_idx    = rand_idx["rand_idx_0"][:(epoch+1)*data_num]
valid_rand_idx = rand_idx["rand_idx_1"][:(epoch+1)*data_num]
#################################################################
# Loss Log
ce_criterion = nn.CrossEntropyLoss(ignore_index=gene_tokenizer.pad_token_id, reduction="mean")
softmax      = nn.Softmax(dim=-1)
log_softmax  = nn.LogSoftmax(dim=-1)

sft_ce_loss_log   = []
train_ce_loss_log = []
valid_ce_loss_log = []
valid_ce_interval = int(replay_num/10)
for replay_itr in range(replay_num):
    ##################################################################################################################################
    # Sample Buffer
    ##################################################################################################################################
    sft_data_buffer   = []
    valid_data_buffer = []
    for step_itr in range(buffer_num):
        sft_data_buffer.append(train_dataset[ce_rand_idx[step_itr+replay_itr*buffer_num]])
        valid_data_buffer.append(train_dataset[valid_rand_idx[step_itr+replay_itr*buffer_num]])
    ##################################################################################################################################
    # Cross Entropy for Validation
    ##################################################################################################################################
    gene_model.eval()
    for step_itr in range(gene_step_num):
        valid_batch = train_dataset.collate_fn(valid_data_buffer[step_itr*batch_size:(step_itr+1)*batch_size])
        valid_joint_mask = torch.where(valid_batch["g_joint_input_ids_padded"]==gene_tokenizer.pad_token_id, 0, 1)
        if src_len_flag==True: valid_source_len = len(valid_batch["g_source_input_ids_padded"][0])
        else                 : valid_source_len = 0
        with torch.no_grad(): valid_output = gene_model(input_ids=valid_batch["g_joint_input_ids_padded"].to(device),
                                                        #position_ids=valid_joint_mask.cumsum(axis=-1).to(device)-1,
                                                        attention_mask=valid_joint_mask.to(device))
        valid_logits = valid_output.logits[:, valid_source_len:-1]
        if src_len_flag==True: valid_loss = ce_criterion(valid_logits.contiguous().view(-1, valid_logits.size(-1)),
                                                         valid_batch["g_target_input_ids_padded"][:, 1:].contiguous().view(-1).to(device))
        else                 : valid_loss = ce_criterion(valid_logits.contiguous().view(-1, valid_logits.size(-1)),
                                                         valid_batch["g_joint_input_ids_padded"][:, 1:].contiguous().view(-1).to(device))
        train_ce_loss_log.append(valid_loss.item())
    
    if replay_itr%valid_ce_interval==0:
        valid_loss = torch.tensor(0., requires_grad=False).to(device)
        for valid_batch in valid_dataloader:
            valid_joint_mask = torch.where(valid_batch["g_joint_input_ids_padded"]==gene_tokenizer.pad_token_id, 0, 1)
            if src_len_flag==True: valid_source_len = len(valid_batch["g_source_input_ids_padded"][0])
            else                 : valid_source_len = 0
            with torch.no_grad(): valid_output = gene_model(input_ids=valid_batch["g_joint_input_ids_padded"].to(device),
                                                            #position_ids=valid_joint_mask.cumsum(axis=-1).to(device)-1,
                                                            attention_mask=valid_joint_mask.to(device))
            valid_logits = valid_output.logits[:, valid_source_len:-1]
            if src_len_flag==True: valid_loss += ce_criterion(valid_logits.contiguous().view(-1, valid_logits.size(-1)),
                                                              valid_batch["g_target_input_ids_padded"][:, 1:].contiguous().view(-1).to(device))
            else                 : valid_loss += ce_criterion(valid_logits.contiguous().view(-1, valid_logits.size(-1)),
                                                              valid_batch["g_joint_input_ids_padded"][:, 1:].contiguous().view(-1).to(device))
        valid_ce_loss_log.append(valid_loss.item() / len(valid_dataloader))
    del valid_batch, valid_joint_mask, valid_source_len, valid_output, valid_logits, valid_loss
    torch.cuda.empty_cache()

    ##################################################################################################################################
    # Print Valid
    ##################################################################################################################################
    if replay_itr>=50*valid_ce_interval:
        if (replay_itr%valid_ce_interval==0 and valid_ce_loss_log[-1]==np.min(valid_ce_loss_log)) \
        or replay_itr==50*valid_ce_interval:
            print("####################################################################################################################################################")
            print("Better MODEL at " + str(replay_itr+1) + "itr | valid_ce: " + str(valid_ce_loss_log[-1]))
            print("####################################################################################################################################################")

    ##################################################################################################################################
    # Plot
    ##################################################################################################################################
    if replay_itr%valid_ce_interval==0:
        print("{}/{} replay_num: {:.2f} mins.".format(replay_itr+1, replay_num, (time.time()-init_time)/60))
        utils.functions.plot_result(dir_path=dir_path,
                                    seed=seed,
                                    replay_itr=replay_itr,
                                    gene_step_num=gene_step_num,
                                    dataset_type=dataset_type,
                                    method_type=method_type,
                                    model_type=model_type,
                                    gene_model_idx=gene_model_idx,
                                    train_ce_loss_log=train_ce_loss_log,
                                    valid_ce_loss_log=valid_ce_loss_log,
                                    valid_ce_interval=valid_ce_interval)
    
    ##################################################################################################################################
    # Supervised Fine Tuning with Cross Entropy
    ##################################################################################################################################
    gene_model.train()
    #################################################################
    # Generator Output
    #################################################################
    # logits : Shape of (batch_size, input_ids_len, vocab_size)
    # probs  : Shape of (batch_size, input_ids_len, vocab_size)
    # gene_joint_ids : <BOS> + source_text + <PAD> | *<SEP> +  target_text + <EOS> + <PAD>
    # output.logits  :         source_text + <PAD> |  <SEP> + *target_text + <EOS> + <PAD> + ?
    #################################################################
    for step_itr in range(gene_step_num):
        sft_batch = train_dataset.collate_fn(sft_data_buffer[step_itr*batch_size:(step_itr+1)*batch_size])
        sft_joint_mask = torch.where(sft_batch["g_joint_input_ids_padded"]==gene_tokenizer.pad_token_id, 0, 1)
        if src_len_flag==True: sft_source_len = len(sft_batch["g_source_input_ids_padded"][0])
        else                 : sft_source_len = 0
        sft_output = gene_model(input_ids=sft_batch["g_joint_input_ids_padded"].to(device),
                                #position_ids=sft_joint_mask.cumsum(axis=-1).to(device)-1,
                                attention_mask=sft_joint_mask.to(device))
        sft_logits = sft_output.logits[:, sft_source_len:-1]
        del sft_output, sft_joint_mask, sft_source_len
        torch.cuda.empty_cache()
        
        #################################################################
        # Cross Entropy Loss
        #################################################################
        if src_len_flag==True: sft_ce_loss = ce_criterion(sft_logits.contiguous().view(-1, sft_logits.size(-1)),
                                                          sft_batch["g_target_input_ids_padded"][:, 1:].contiguous().view(-1).to(device))
        else                 : sft_ce_loss = ce_criterion(sft_logits.contiguous().view(-1, sft_logits.size(-1)),
                                                          sft_batch["g_joint_input_ids_padded"][:, 1:].contiguous().view(-1).to(device))
        sft_ce_loss_log.append(sft_ce_loss.item())
        gene_loss = sft_ce_loss
        gene_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(gene_model.parameters(), max_gradient_norm)
        gene_optimizer.step()
        gene_scheduler.step()
        gene_optimizer.zero_grad()
        del sft_batch, sft_logits, sft_ce_loss, gene_loss
        torch.cuda.empty_cache()

##################################################################################################################################
# Plot
##################################################################################################################################
print("{}/{} replay_num: {:.2f} mins.".format(replay_itr+1, replay_num, (time.time()-init_time)/60))
utils.functions.plot_result(dir_path=dir_path,
                            seed=seed,
                            replay_itr=replay_itr,
                            gene_step_num=gene_step_num,
                            dataset_type=dataset_type,
                            method_type=method_type,
                            model_type=model_type,
                            gene_model_idx=gene_model_idx,
                            train_ce_loss_log=train_ce_loss_log,
                            valid_ce_loss_log=valid_ce_loss_log,
                            valid_ce_interval=valid_ce_interval)

#################################################################
# Save Model
#################################################################
if gene_model_idx>=0:
    torch.save(gene_model.state_dict(), model_path + dataset_type + "/model/" + method_type + "/gene_model_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx.pth")
    torch.save({"sft_ce_loss_log"  : sft_ce_loss_log,
                "train_ce_loss_log": train_ce_loss_log,
                "valid_ce_loss_log": valid_ce_loss_log},
                dir_path + dataset_type + "/data/" + method_type + "/loss_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx.pth")
else:
    torch.save(gene_model.state_dict(), model_path + dataset_type + "/model/" + method_type + "/gene_model_" + str(model_type) + "_" + str(seed) + "seed.pth")
    torch.save({"sft_ce_loss_log"  : sft_ce_loss_log,
                "train_ce_loss_log": train_ce_loss_log,
                "valid_ce_loss_log": valid_ce_loss_log},
                dir_path + dataset_type + "/data/" + method_type + "/loss_" + str(model_type) + "_" + str(seed) + "seed.pth")

if valid_ce_loss_log[-1]==np.min(valid_ce_loss_log):
    print("####################################################################################################################################################")
    print("BEST MODEL at LAST itr | valid_ce: " + str(valid_ce_loss_log[-1]))
print("####################################################################################################################################################")
print("ALL WORK Finished: {:.2f} mins.".format((time.time()-init_time)/60))
print("####################################################################################################################################################")
