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

from transformers import GPT2Tokenizer,  GPT2LMHeadModel  #355M
from transformers import AutoTokenizer,  OPTForCausalLM   #350M #arxiv:2205.01068 "facebook/opt-350m"
#from transformers import AutoTokenizer,  BloomForCausalLM #560M #arxiv:2211.01786 "bigscience/bloomz-560m"

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
parser.add_argument("--decode_type",        type=str,   default="top-p")
parser.add_argument("--reward_type",        type=str,   default="confid")
parser.add_argument("--norm_type",          type=str,   default="robust_scaler")
parser.add_argument("--clip_type",          type=str,   default="ppo")
parser.add_argument("--real_reward_inter",  type=float, default=2.0)
parser.add_argument("--gene_reward_inter",  type=float, default=0.0)
parser.add_argument("--gene_learning_rate", type=float, default=1.0e-6)
parser.add_argument("--disc_learning_rate", type=float, default=1.0e-7)
parser.add_argument("--amdail_alpha",       type=float, default=0.8)
parser.add_argument("--amdail_beta",        type=float, default=0.2)
parser.add_argument("--ppo_epsilon",        type=float, default=0.10)
parser.add_argument("--ppo_epsilon_gene",   type=float, default=0.10)
parser.add_argument("--gene_model_idx",     type=int,   default=0)
parser.add_argument("--disc_model_idx",     type=int,   default=0)
parser.add_argument("--model_mode",         type=str,   default="train")
args               = parser.parse_args()
seed               = args.seed
dataset_type       = args.dataset_type
model_type         = args.model_type
method_type        = args.method_type
decode_type        = args.decode_type
reward_type        = args.reward_type
norm_type          = args.norm_type
clip_type          = args.clip_type
real_reward_inter  = args.real_reward_inter
gene_reward_inter  = args.gene_reward_inter

gene_learning_rate = args.gene_learning_rate
disc_learning_rate = args.disc_learning_rate

amdail_alpha       = args.amdail_alpha
amdail_beta        = args.amdail_beta
ppo_epsilon        = args.ppo_epsilon
ppo_epsilon_gene   = args.ppo_epsilon_gene

gene_model_idx     = args.gene_model_idx
disc_model_idx     = args.disc_model_idx
model_mode         = args.model_mode

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
if not os.path.exists(dir_path   + dataset_type + "/data/"   + reward_type + "_" + clip_type): os.makedirs(dir_path   + dataset_type + "/data/"   + reward_type + "_" + clip_type)
if not os.path.exists(dir_path   + dataset_type + "/figure/" + reward_type + "_" + clip_type): os.makedirs(dir_path   + dataset_type + "/figure/" + reward_type + "_" + clip_type)
if not os.path.exists(model_path + dataset_type + "/model/"  + reward_type + "_" + clip_type): os.makedirs(model_path + dataset_type + "/model/"  + reward_type + "_" + clip_type)

with open(dir_path + "config/" + dataset_type + "_" + str(model_type) + ".yaml", "r") as f: config = yaml.load(f, Loader=yaml.Loader)[method_type]
src_len_flag       = config["src_len_flag"]
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
ppo_real_alpha     = config["ppo_real_alpha"]
ppo_gene_alpha     = config["ppo_gene_alpha"]
kld_alpha          = 2.0

print("==================================================================================")
print("seed        : " + str(seed))
print("dataset_type: " + str(dataset_type))
print("model_type  : " + model_type)
print("method_type : " + method_type)
print("reward_type : " + reward_type)
print("clip_type   : " + clip_type)
print("==================================================================================")
print("epoch       : " + str(epoch))
print("batch_size  : " + str(batch_size))
print("top_p       : " + str(top_p))
print("gene_learning_rate : " + str(gene_learning_rate))
print("disc_learning_rate : " + str(disc_learning_rate))
print("weight_decay       : " + str(weight_decay))
print("real_reward_inter  : " + str(real_reward_inter))
print("gene_reward_inter  : " + str(gene_reward_inter))
print("gene_model_idx     : " + str(gene_model_idx))
print("disc_model_idx     : " + str(disc_model_idx))
print("model_mode         : " + str(model_mode))
print("ppo_epsilon        : " + str(ppo_epsilon))
if reward_type=="amdail":
    print("amdail_alpha       : " + str(amdail_alpha))
    print("amdail_beta        : " + str(amdail_beta))
if clip_type=="amdail":
    print("amdail_alpha       : " + str(amdail_alpha))
    print("ppo_epsilon_gene   : " + str(ppo_epsilon_gene))
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
if model_type=="gpt2"  : gene_tokenizer =  GPT2Tokenizer.from_pretrained("gpt2-medium",
                                                                         add_special_tokens=False,
                                                                         sep_token=disc_tokenizer.sep_token,
                                                                         pad_token=disc_tokenizer.pad_token,
                                                                         bos_token=temp_tokenizer.bos_token,
                                                                         eos_token=temp_tokenizer.eos_token)
elif model_type=="opt" : gene_tokenizer =  AutoTokenizer.from_pretrained("facebook/opt-350m",
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

#################################################################
# Dataset / Reset random seed
# "common_gen"
# "adamlin/roc_story"
#################################################################
if dataset_type=="common_gen" : dataset = load_dataset(dataset_type)
elif dataset_type=="roc_story": dataset = load_dataset("csv", data_files={"train"     : dir_path + "dataset/roc_train_" + str(seed) + "seed.csv",
                                                                          "validation": dir_path + "dataset/roc_valid_" + str(seed) + "seed.csv"})
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

disc_batch_size     = int(batch_size/2)
disc_step_num       = int(buffer_num/disc_batch_size)
disc_training_steps = replay_num*disc_step_num
disc_warmup_steps   = int(0.2*replay_num*disc_step_num)
print("gene_training_steps: " + str(gene_training_steps))
print("gene_warmup_steps  : " + str(gene_warmup_steps))
print("disc_training_steps: " + str(disc_training_steps))
print("disc_warmup_steps  : " + str(disc_warmup_steps))
print("==================================================================================")

#################################################################
# Instance Generator and Optimizer
#################################################################
if model_type=="gpt2"   : gene_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
elif model_type=="opt"  : gene_model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
elif model_type=="bloom": gene_model = BloomForCausalLM.from_pretrained("bigscience/bloomz-560m")
#################################################################
gene_model.resize_token_embeddings(len(gene_tokenizer))
if model_type=="bart": gene_model.config.is_encoder_decoder = False
gene_model.layer_norm_epsilon = layer_norm_epsilon
gene_model.temperature = temperature
for p in gene_model.parameters(): p.grad = None
#################################################################
gene_optimizer = torch.optim.AdamW(gene_model.parameters(),
                                   lr=gene_learning_rate,
                                   weight_decay=weight_decay,
                                   maximize=True)
#################################################################
if model_type=="gpt2"   : pre_gene_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
elif model_type=="opt"  : pre_gene_model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
elif model_type=="bloom": pre_gene_model = BloomForCausalLM.from_pretrained("bigscience/bloomz-560m")
#################################################################
pre_gene_model.resize_token_embeddings(len(gene_tokenizer))
if model_type=="bart": pre_gene_model.config.is_encoder_decoder = False
pre_gene_model.layer_norm_epsilon = layer_norm_epsilon
pre_gene_model.temperature = temperature
for p in pre_gene_model.parameters(): p.grad = None
#################################################################
class DebertaClassifier(nn.Module):
    def __init__(self):
        super(DebertaClassifier, self).__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.linear = nn.Linear(self.deberta.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask=None):
        output = self.deberta(input_ids=input_ids,
                              attention_mask=attention_mask)
        output = output.last_hidden_state[:, 0, :]
        output = self.linear(output)
        return output
#################################################################
disc_model = DebertaClassifier()
for p in disc_model.parameters(): p.grad = None
disc_optimizer = torch.optim.AdamW(disc_model.parameters(),
                                   lr=disc_learning_rate,
                                   weight_decay=weight_decay,
                                   maximize=True)
#################################################################
if model_mode=="train" : pre_gene_model.train()
elif model_mode=="eval": pre_gene_model.eval()
gene_model.train()
disc_model.train()
#################################################################
print("gene_model_size: " + str(int(gene_model.num_parameters()*1e-6))         + "M parameters")
print("disc_model_size: " + str(int(disc_model.deberta.num_parameters()*1e-6)) + "M parameters")
print("==================================================================================")

#################################################################
# Scheduler and Loss Fucntion
#################################################################
gene_scheduler = get_linear_schedule_with_warmup(gene_optimizer,
                                                 num_warmup_steps=gene_warmup_steps,
                                                 num_training_steps=gene_training_steps)
disc_scheduler = get_linear_schedule_with_warmup(disc_optimizer,
                                                 num_warmup_steps=disc_warmup_steps,
                                                 num_training_steps=disc_training_steps)
gene_model, pre_gene_model, disc_model, gene_optimizer, disc_optimizer = \
    accelerator.prepare(gene_model, pre_gene_model, disc_model, gene_optimizer, disc_optimizer)

#################################################################
# Train
#################################################################
rand_idx = torch.load(dir_path + dataset_type + "/data/rand_idx_" + str(seed) +  "seed.pth")
sample_rand_idx = rand_idx["rand_idx_3"][:(epoch+1)*data_num]
if gene_model_idx>=0:
    gene_model.load_state_dict(torch.load(model_path     + dataset_type + "/model/sft/gene_model_"  + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx%10) + "idx.pth"))
    pre_gene_model.load_state_dict(torch.load(model_path + dataset_type + "/model/sft/gene_model_"  + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx%10) + "idx.pth"))
    disc_model.load_state_dict(torch.load(model_path     + dataset_type + "/model/disc/disc_model_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx%10) + "idx_" + str(disc_model_idx%10) + "idx.pth"))
else:
    gene_model.load_state_dict(torch.load(model_path     + dataset_type + "/model/sft/gene_model_"  + str(model_type) + "_" + str(seed) + "seed.pth"))
    pre_gene_model.load_state_dict(torch.load(model_path + dataset_type + "/model/sft/gene_model_"  + str(model_type) + "_" + str(seed) + "seed.pth"))
    disc_model.load_state_dict(torch.load(model_path     + dataset_type + "/model/disc/disc_model_" + str(model_type) + "_" + str(seed) + "seed.pth"))
for p in gene_model.parameters()    : p.grad = None
for p in pre_gene_model.parameters(): p.grad = None
for p in disc_model.parameters()    : p.grad = None

#################################################################
# Loss Log
ce_criterion = nn.CrossEntropyLoss(ignore_index=gene_tokenizer.pad_token_id, reduction="mean")
softmax      = nn.Softmax(dim=-1)
log_softmax  = nn.LogSoftmax(dim=-1)

valid_ce_interval = int(replay_num/10)

bleu_score = evaluate.load("google_bleu")
ppo_bleu_score_log     = []
ppo_distinct_score_log = []

disc_loss_log     = []
ppo_gene_loss_log = []
sft_real_loss_log = []
kld_gene_loss_log = []
gene_loss_log     = []

confid_gene_mean_log   = []
confid_gene_median_log = []
confid_gene_upperQ_log = []
confid_gene_lowerQ_log = []
confid_gene_max_log    = []
confid_gene_min_log    = []

ppo_gene_ratio_log         = []
ppo_gene_ratio_clipped_log = []
ppo_gene_confid_log        = []

print("####################################################################################################################################################")
for replay_itr in range(replay_num):
    ##################################################################################################################################
    # Sample Buffer
    ##################################################################################################################################
    ppo_real_data_buffer = []
    ppo_real_text_buffer = []
    ppo_gene_text_buffer = []
    ppo_real_confid_buffer = []
    ppo_gene_confid_buffer = []
    for step_itr in range(buffer_num): ppo_real_data_buffer.append(train_dataset[sample_rand_idx[step_itr+replay_itr*buffer_num]])    
    ##################################################################################################################################
    # Plot
    ##################################################################################################################################
    if replay_itr%valid_ce_interval==0:
        print("{}/{} replay_num: {:.2f} mins.".format(replay_itr+1, replay_num, (time.time()-init_time)/60))
        if replay_itr!=0:
            utils.functions.plot_ratio(seed=seed,
                                       dir_path=dir_path,
                                       dataset_type=dataset_type,
                                       model_type=model_type,
                                       model_idx=gene_model_idx,
                                       reward_type=reward_type,
                                       clip_type=clip_type,
                                       ppo_gene_ratio_log=ppo_gene_ratio_log[-10*buffer_num:],
                                       ppo_gene_ratio_clipped_log=ppo_gene_ratio_clipped_log[-10*buffer_num:],
                                       ppo_gene_confid_log=ppo_gene_confid_log[-10*buffer_num:])
            utils.functions.plot_result(replay_itr=replay_itr,
                                        gene_step_num=gene_step_num,
                                        valid_ce_interval=valid_ce_interval,
                                        seed=seed,
                                        dir_path=dir_path,
                                        dataset_type=dataset_type,
                                        model_type=model_type,
                                        gene_model_idx=gene_model_idx,
                                        disc_model_idx=disc_model_idx,
                                        method_type=method_type,
                                        reward_type=reward_type,
                                        clip_type=clip_type,
                                        ppo_bleu_score_log=ppo_bleu_score_log,
                                        ppo_distinct_score_log=ppo_distinct_score_log,
                                        confid_gene_upperQ_log=confid_gene_upperQ_log,
                                        confid_gene_lowerQ_log=confid_gene_lowerQ_log,
                                        confid_gene_max_log=confid_gene_max_log,
                                        confid_gene_min_log=confid_gene_min_log,
                                        confid_gene_median_log=confid_gene_median_log,
                                        ppo_gene_loss_log=ppo_gene_loss_log,
                                        sft_real_loss_log=sft_real_loss_log,
                                        kld_gene_loss_log=kld_gene_loss_log)
    
    ##################################################################################################################################
    # Buffer Generated, Real Text, Confidence, and Reward
    ##################################################################################################################################
    # Buffer Generated and Real Text
    # str -> gene_tokenizer -> id | id-> gene_generator -> id
    #################################################################
    if model_mode=="train" : gene_model.train()
    elif model_mode=="eval": gene_model.eval()
    #################################################################
    for step_itr in range(gene_step_num):
        #################################################################
        # Decode Sentence
        #################################################################
        ppo_gene_batch = train_dataset.collate_fn(ppo_real_data_buffer[step_itr*batch_size:(step_itr+1)*batch_size])
        ppo_real_batch = train_dataset.collate_fn(ppo_real_data_buffer[step_itr*batch_size:(step_itr+1)*batch_size])
        if model_type=="gpt2": ppo_gene_source_input_ids = torch.cat((ppo_gene_batch["g_source_input_ids_padded"], \
                                                                      gene_tokenizer.sep_token_id * torch.ones((batch_size, 1), dtype=torch.long)),
                                                                      dim=1)
        else: ppo_gene_source_input_ids = torch.cat((ppo_gene_batch["g_source_input_ids_padded"], \
                                                     gene_tokenizer(" ", add_special_tokens=False)["input_ids"][0] * torch.ones((batch_size, 1), dtype=torch.long), \
                                                     gene_tokenizer.sep_token_id * torch.ones((batch_size, 1), dtype=torch.long)),
                                                     dim=1)
        ppo_gene_source_mask = torch.where(ppo_gene_source_input_ids==gene_tokenizer.pad_token_id, 0, 1)
        
        with torch.no_grad():
            if decode_type=="top-p": ppo_gene_joint_input_ids = gene_model.generate(ppo_gene_source_input_ids.to(device),
                                                                                    max_length=max_length,
                                                                                    do_sample=True,
                                                                                    early_stopping=True,
                                                                                    temperature=temperature,
                                                                                    top_p=top_p,
                                                                                    repetition_penalty=1.0,
                                                                                    pad_token_id=gene_tokenizer.pad_token_id,
                                                                                    bos_token_id=gene_tokenizer.bos_token_id,
                                                                                    eos_token_id=gene_tokenizer.eos_token_id,
                                                                                    length_penalty=1.0,
                                                                                    attention_mask=ppo_gene_source_mask.to(device),
                                                                                    remove_invalid_values=True)
            elif decode_type=="beam": ppo_gene_joint_input_ids = gene_model.generate(ppo_gene_source_input_ids.to(device),
                                                                                     max_length=max_length,
                                                                                     num_beams=4,
                                                                                     do_sample=True,
                                                                                     early_stopping=True,
                                                                                     temperature=temperature,
                                                                                     repetition_penalty=1.0,
                                                                                     pad_token_id=gene_tokenizer.pad_token_id,
                                                                                     bos_token_id=gene_tokenizer.bos_token_id,
                                                                                     eos_token_id=gene_tokenizer.eos_token_id,
                                                                                     length_penalty=1.0,
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
        del ppo_real_joint_text_, ppo_gene_joint_text_, \
            ppo_real_joint_text,  ppo_gene_joint_text
        torch.cuda.empty_cache()
    #################################################################
    # Calculate BLEU, Distinct, and Bert Score
    #################################################################
    bleu_real_text = []
    bleu_ppo_text  = []
    for item in ppo_real_text_buffer:
        item = item.split(gene_tokenizer.sep_token)[1]
        item = item + " " + gene_tokenizer.eos_token
        item = item.split(gene_tokenizer.eos_token)[0]
        item = item.strip()
        bleu_real_text.append(item)
    for item in ppo_gene_text_buffer:
        item = item.split(gene_tokenizer.sep_token)[1]
        item = item + " " + gene_tokenizer.eos_token
        item = item.split(gene_tokenizer.eos_token)[0]
        item = item.strip()
        if item in ["", gene_tokenizer.pad_token]: item = gene_tokenizer.pad_token + " " + gene_tokenizer.pad_token
        bleu_ppo_text.append(item)
    ppo_bleu_score_list = []
    for score_itr in range(buffer_num): ppo_bleu_score_list.append(bleu_score.compute(predictions=[bleu_ppo_text[score_itr]], references=[[bleu_real_text[score_itr]]])["google_bleu"])
    ppo_bleu_score_log.append(np.mean(ppo_bleu_score_list))
    ppo_distinct_score_log.append(utils.functions.compute_distinct(input=bleu_ppo_text, n=2))

    if replay_itr%valid_ce_interval==0:
        print("bleu_real  : " + repr(bleu_real_text[0]))
        print("bleu_ppo   : " + repr(bleu_ppo_text[0]))
        print("sample_real: " + repr(ppo_real_text_buffer[0]))
        print("sample_ppo : " + repr(ppo_gene_text_buffer[0]))
        print("====================================================================================================================================================")
        #with redirect_stdout(open(os.devnull, "w")): bert_score_list = bert_score.compute(predictions=bleu_gene_text, references=bleu_real_text, method_type="microsoft/deberta-xlarge-mnli")["f1"]
        #bert_score_log.append(np.mean(bert_score_list))
    
    #################################################################
    # Buffer Confidence
    #################################################################
    ppo_gene_text_buffer, ppo_real_text_buffer = utils.functions.shuffle_pair(gene_text_buffer=ppo_gene_text_buffer,
                                                                              real_text_buffer=ppo_real_text_buffer)
    #################################################################
    if model_mode=="train" : disc_model.train()
    elif model_mode=="eval": disc_model.eval()
    #################################################################
    for step_itr in range(disc_step_num):
        ppo_joint_text_ = []
        ppo_joint_text_.extend(ppo_real_text_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        ppo_joint_text_.extend(ppo_gene_text_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        
        ppo_joint_text = []
        for item in ppo_joint_text_:
            item = item.replace(gene_tokenizer.sep_token, disc_tokenizer.sep_token, 1)
            ppo_joint_text.append(item)
        del ppo_joint_text_
        torch.cuda.empty_cache()
        #################################################################
        #################################################################
        #################################################################
        mix_source_batch = []
        mix_target_batch = []
        for item in ppo_joint_text:
            item    = item.split(gene_tokenizer.sep_token, maxsplit=1)
            item[0] = item[0].strip()
            item[1] = " " + gene_tokenizer.sep_token + item[1]
            mix_source_batch.append(item[0])
            mix_target_batch.append(item[1])
        mix_source_input_ids = gene_tokenizer(mix_source_batch,
                                              padding=True,
                                              return_tensors="pt",
                                              add_special_tokens=False)["input_ids"]
        mix_target_input_ids = gene_tokenizer(mix_target_batch,
                                              padding=True,
                                              return_tensors="pt",
                                              add_special_tokens=False)["input_ids"]
        mix_joint_input_ids  = torch.cat((mix_source_input_ids, mix_target_input_ids), dim=1)
        ppo_joint_input_ids  = mix_joint_input_ids
        #################################################################
        #################################################################
        #################################################################
        ppo_joint_mask = torch.where(ppo_joint_input_ids==disc_tokenizer.pad_token_id, 0, 1)
        del ppo_joint_text
        torch.cuda.empty_cache()

        with torch.no_grad(): disc_logit = disc_model(ppo_joint_input_ids.to(device),
                                                      attention_mask=ppo_joint_mask.to(device))
        real_logit = disc_logit[:disc_batch_size]
        gene_logit = disc_logit[disc_batch_size:]
        
        ppo_real_confid = log_softmax(torch.cat((real_logit[:, 0].unsqueeze(-1), gene_logit[:, 0].unsqueeze(-1)), dim=1))[:, 0].exp()
        ppo_gene_confid = log_softmax(torch.cat((real_logit[:, 0].unsqueeze(-1), gene_logit[:, 0].unsqueeze(-1)), dim=1))[:, 1].exp()
        ppo_real_confid_buffer.extend(ppo_real_confid.tolist())
        ppo_gene_confid_buffer.extend(ppo_gene_confid.tolist())
        del ppo_joint_input_ids, ppo_joint_mask, disc_logit, real_logit, gene_logit, \
            ppo_real_confid, ppo_gene_confid
        torch.cuda.empty_cache()
    confid_gene_mean_log.append(np.mean(ppo_gene_confid_buffer))
    confid_gene_median_log.append(np.median(ppo_gene_confid_buffer))
    confid_gene_upperQ_log.append(np.percentile(ppo_gene_confid_buffer, q=75))
    confid_gene_lowerQ_log.append(np.percentile(ppo_gene_confid_buffer, q=25))
    confid_gene_max_log.append(np.max(ppo_gene_confid_buffer))
    confid_gene_min_log.append(np.min(ppo_gene_confid_buffer))
    
    ##################################################################################################################################
    # GAN Training
    ##################################################################################################################################
    # Discriminator Training
    ##################################################################################################################################
    disc_gene_text_buffer, disc_real_text_buffer = utils.functions.shuffle_pair(gene_text_buffer=ppo_gene_text_buffer,
                                                                                real_text_buffer=ppo_real_text_buffer)
    #################################################################
    disc_model.train()
    #################################################################
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
        #################################################################
        #################################################################
        #################################################################
        disc_source_batch = []
        disc_target_batch = []
        for item in disc_joint_text:
            item    = item.split(gene_tokenizer.sep_token, maxsplit=1)
            item[0] = item[0].strip()
            item[1] = " " + gene_tokenizer.sep_token + item[1]
            disc_source_batch.append(item[0])
            disc_target_batch.append(item[1])
        disc_source_input_ids = gene_tokenizer(disc_source_batch,
                                               padding=True,
                                               return_tensors="pt",
                                               add_special_tokens=False)["input_ids"]
        disc_target_input_ids = gene_tokenizer(disc_target_batch,
                                               padding=True,
                                               return_tensors="pt",
                                               add_special_tokens=False)["input_ids"]
        disc_joint_input_ids  = torch.cat((disc_source_input_ids, disc_target_input_ids), dim=1)
        #################################################################
        #################################################################
        #################################################################
        disc_joint_mask = torch.where(disc_joint_input_ids==disc_tokenizer.pad_token_id, 0, 1)
        del disc_joint_text_, disc_joint_text
        torch.cuda.empty_cache()

        disc_logit = disc_model(disc_joint_input_ids.to(device),
                                attention_mask=disc_joint_mask.to(device))
        real_logit = disc_logit[:disc_batch_size]
        gene_logit = disc_logit[disc_batch_size:]
        
        real_confid = log_softmax(torch.cat((real_logit[:, 0].unsqueeze(-1), gene_logit[:, 0].unsqueeze(-1)), dim=1))[:, 0]
        gene_confid = log_softmax(torch.cat((real_logit[:, 1].unsqueeze(-1), gene_logit[:, 1].unsqueeze(-1)), dim=1))[:, 1]
        del disc_joint_input_ids, disc_joint_mask, disc_logit, real_logit, gene_logit
        torch.cuda.empty_cache()

        #################################################################
        # Discriminator Loss
        #################################################################
        disc_loss = disc_alpha * (real_confid[:].sum() + gene_confid[:].sum()) / batch_size
        disc_loss_log.append(disc_loss.item())
        del real_confid, gene_confid
        torch.cuda.empty_cache()
        
        disc_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(disc_model.parameters(), max_gradient_norm)
        disc_optimizer.step()
        disc_scheduler.step()
        disc_optimizer.zero_grad()
        del disc_loss
        torch.cuda.empty_cache()
    
    ##################################################################################################################################
    # Generator Training
    # Calculate Reward
    ##################################################################################################################################
    ppo_real_confid_buffer = torch.tensor(ppo_real_confid_buffer)
    ppo_gene_confid_buffer = torch.tensor(ppo_gene_confid_buffer)
    ppo_real_reward_buffer = real_reward_inter * torch.ones(buffer_num)
    
    if reward_type=="confid"  : ppo_gene_reward_buffer = ppo_gene_confid_buffer
    elif reward_type=="gail"  : ppo_gene_reward_buffer = ppo_gene_confid_buffer
    elif reward_type=="airl"  : ppo_gene_reward_buffer = torch.log(ppo_gene_confid_buffer) - torch.log(1-ppo_gene_confid_buffer)
    elif reward_type=="amdail": ppo_gene_reward_buffer = (amdail_alpha*(1-ppo_gene_confid_buffer) + (1-amdail_alpha)*ppo_gene_confid_buffer) / (1-ppo_gene_confid_buffer) \
                                                         * torch.log((amdail_beta *(1-ppo_gene_confid_buffer) + (1-amdail_beta) *ppo_gene_confid_buffer) \
                                                                   / (amdail_alpha*(1-ppo_gene_confid_buffer) + (1-amdail_alpha)*ppo_gene_confid_buffer))
    
    if reward_type=="zero": ppo_gene_reward_buffer = torch.zeros(buffer_num)
    else:
        if norm_type=="robust_scaler":
            ppo_gene_reward_buffer = ppo_gene_reward_buffer - torch.median(ppo_gene_reward_buffer)
            ppo_gene_reward_buffer = 1.3489 * ppo_gene_reward_buffer \
                                   / (torch.quantile(input=ppo_gene_reward_buffer, q=0.75)-torch.quantile(input=ppo_gene_reward_buffer, q=0.25))
        elif norm_type=="standard_scaler":
            ppo_gene_reward_buffer = ppo_gene_reward_buffer - torch.mean(ppo_gene_reward_buffer)
            ppo_gene_reward_buffer = ppo_gene_reward_buffer / torch.std(ppo_gene_reward_buffer)        
        ppo_gene_reward_buffer = ppo_gene_reward_buffer + gene_reward_inter
        if reward_type=="clip": ppo_gene_reward_buffer = torch.where(ppo_gene_reward_buffer>0.0, ppo_gene_reward_buffer, torch.zeros(buffer_num))
    if replay_itr%valid_ce_interval==0: utils.functions.hist_reward(seed=seed,
                                                                    dir_path=dir_path,
                                                                    dataset_type=dataset_type,
                                                                    model_type=model_type,
                                                                    gene_model_idx=gene_model_idx,
                                                                    disc_model_idx=disc_model_idx,
                                                                    reward_type=reward_type,
                                                                    clip_type=clip_type,
                                                                    gene_reward_buffer=ppo_gene_reward_buffer.tolist())
    ppo_real_confid_buffer = ppo_real_confid_buffer.tolist()
    ppo_real_reward_buffer = ppo_real_reward_buffer.tolist()
    ppo_gene_confid_buffer = ppo_gene_confid_buffer.tolist()
    ppo_gene_reward_buffer = ppo_gene_reward_buffer.tolist()
    
    ##################################################################################################################################
    # Calculate Policy Loss
    ##################################################################################################################################
    ppo_real_text_buffer, ppo_real_reward_buffer, ppo_real_confid_buffer = utils.functions.shuffle_buffer(text_buffer=ppo_real_text_buffer, reward_buffer=ppo_real_reward_buffer, confid_buffer=ppo_real_confid_buffer)
    ppo_gene_text_buffer, ppo_gene_reward_buffer, ppo_gene_confid_buffer = utils.functions.shuffle_buffer(text_buffer=ppo_gene_text_buffer, reward_buffer=ppo_gene_reward_buffer, confid_buffer=ppo_gene_confid_buffer)
    #################################################################
    if clip_type=="amdail":
        D = torch.median(torch.tensor(ppo_gene_confid_buffer))
        A = (1-amdail_alpha) * D + amdail_alpha * (1-D)
        B = (1-amdail_alpha) * D
        C = amdail_alpha * (1-D)
        ppo_epsilon_gene = 1 - (B + C * (1-ppo_epsilon)) / A
    #################################################################
    pre_gene_model.load_state_dict(gene_model.state_dict())
    for p in pre_gene_model.parameters(): p.grad = None
    gene_model.train()
    pre_gene_model.train()
    #################################################################
    ppo_gene_loss_list = []
    for step_itr in range(disc_step_num):
        #################################################################
        gene_loss = torch.tensor(0., requires_grad=True)
        #################################################################
        ppo_mix_text_batch   = []
        ppo_mix_text_batch.extend(ppo_real_text_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        ppo_mix_text_batch.extend(ppo_gene_text_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        ppo_mix_reward_batch = []
        ppo_mix_reward_batch.extend(ppo_real_reward_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        ppo_mix_reward_batch.extend(ppo_gene_reward_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        ppo_mix_confid_batch = []
        ppo_mix_confid_batch.extend(ppo_real_confid_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        ppo_mix_confid_batch.extend(ppo_gene_confid_buffer[step_itr*disc_batch_size:(step_itr+1)*disc_batch_size])
        #################################################################
        ppo_mix_reward_batch = torch.tensor(ppo_mix_reward_batch)
        ppo_mix_confid_batch = torch.tensor(ppo_mix_confid_batch)
        #################################################################
        ppo_mix_source_batch = []
        ppo_mix_target_batch = []
        for item in ppo_mix_text_batch:
            item    = item.split(gene_tokenizer.sep_token, maxsplit=1)
            item[0] = item[0].strip()
            item[1] = " " + gene_tokenizer.sep_token + item[1]
            ppo_mix_source_batch.append(item[0])
            ppo_mix_target_batch.append(item[1])
        ppo_mix_source_input_ids = gene_tokenizer(ppo_mix_source_batch,
                                                  padding=True,
                                                  return_tensors="pt",
                                                  add_special_tokens=False)["input_ids"]
        ppo_mix_target_input_ids = gene_tokenizer(ppo_mix_target_batch,
                                                  padding=True,
                                                  return_tensors="pt",
                                                  add_special_tokens=False)["input_ids"]
        
        if src_len_flag==True: ppo_mix_source_len = len(ppo_mix_source_input_ids[0])
        else                 : ppo_mix_source_len = 0
        ppo_mix_joint_input_ids = torch.cat((ppo_mix_source_input_ids, ppo_mix_target_input_ids), dim=1)
        ppo_mix_joint_mask  = torch.where(ppo_mix_joint_input_ids ==gene_tokenizer.pad_token_id, 0, 1)
        ppo_mix_target_mask = torch.where(ppo_mix_target_input_ids==gene_tokenizer.pad_token_id, 0, 1)
        del ppo_mix_text_batch, ppo_mix_source_batch, ppo_mix_target_batch, ppo_mix_source_input_ids
        torch.cuda.empty_cache()
        #################################################################
        # Post/Pre Generator Output of Texts for PPO
        #################################################################
        ppo_mix_output_of_post_model = gene_model(input_ids=ppo_mix_joint_input_ids.to(device),
                                                  attention_mask=ppo_mix_joint_mask.to(device))
        ppo_mix_output_logit_of_post_model = ppo_mix_output_of_post_model.logits[:, ppo_mix_source_len:-1]
        ppo_mix_log_probs_of_post_model    = log_softmax(ppo_mix_output_logit_of_post_model)
        del ppo_mix_output_of_post_model
        torch.cuda.empty_cache()
        #################################################################
        with torch.no_grad():
            ppo_mix_output_of_pre_model = pre_gene_model(input_ids=ppo_mix_joint_input_ids.to(device),
                                                         attention_mask=ppo_mix_joint_mask.to(device))
            ppo_mix_output_logit_of_pre_model = ppo_mix_output_of_pre_model.logits[:, ppo_mix_source_len:-1].detach().clone()
            ppo_mix_log_probs_of_pre_model    = log_softmax(ppo_mix_output_logit_of_pre_model)
        del ppo_mix_output_of_pre_model, ppo_mix_output_logit_of_pre_model
        torch.cuda.empty_cache()
        #################################################################
        # Policy Loss
        #################################################################
        if src_len_flag==True:
            ppo_mix_log_probs_of_post_model = torch.gather(ppo_mix_log_probs_of_post_model, -1, ppo_mix_target_input_ids[:, 1:].unsqueeze(dim=-1).to(device)).squeeze(-1)
            ppo_mix_log_probs_of_pre_model  = torch.gather(ppo_mix_log_probs_of_pre_model,  -1, ppo_mix_target_input_ids[:, 1:].unsqueeze(dim=-1).to(device)).squeeze(-1)
        else:
            ppo_mix_log_probs_of_post_model = torch.gather(ppo_mix_log_probs_of_post_model, -1, ppo_mix_joint_input_ids[:, 1:].unsqueeze(dim=-1).to(device)).squeeze(-1)
            ppo_mix_log_probs_of_pre_model  = torch.gather(ppo_mix_log_probs_of_pre_model,  -1, ppo_mix_joint_input_ids[:, 1:].unsqueeze(dim=-1).to(device)).squeeze(-1)
        
        #################################################################
        # PPO Loss
        #################################################################
        ratio_flag = True
        #################################################################        
        ppo_mix_ratio_ = ppo_mix_log_probs_of_post_model[:] - ppo_mix_log_probs_of_pre_model[:]
        if ratio_flag==False: ppo_mix_ratio_ = ppo_mix_ratio_[:].clamp(min=-16.0, max=16.0)
        if src_len_flag==True:
            ppo_mix_ratio  = torch.where(ppo_mix_target_mask[:, 1:].to(device)==1,
                                         ppo_mix_ratio_[:],
                                         torch.zeros((batch_size, len(ppo_mix_ratio_[0, :])), device=device))
            if ratio_flag==True: ppo_mix_ratio = (ppo_mix_ratio.sum(dim=-1) / ppo_mix_target_mask[:, 1:].sum(dim=-1).to(device)).exp()
        else:
            ppo_mix_ratio  = torch.where(ppo_mix_joint_mask[:, 1:].to(device)==1,
                                         ppo_mix_ratio_[:],
                                         torch.zeros((batch_size, len(ppo_mix_ratio_[0, :])), device=device))
            if ratio_flag==True: ppo_mix_ratio = (ppo_mix_ratio.sum(dim=-1) / ppo_mix_joint_mask[:, 1:].sum(dim=-1).to(device)).exp()
        if ratio_flag==False: ppo_mix_ratio = ppo_mix_ratio.exp()
        del ppo_mix_log_probs_of_post_model, ppo_mix_log_probs_of_pre_model, ppo_mix_ratio_
        torch.cuda.empty_cache()
        #################################################################
        #################################################################
        if clip_type=="ppo"     : clip_min, clip_max = 1.0-ppo_epsilon, 1.0+ppo_epsilon
        elif clip_type=="amdail": clip_min, clip_max = ((amdail_alpha*(1-ppo_mix_confid_batch[disc_batch_size:].to(device)) + (1-amdail_alpha)*ppo_mix_confid_batch[disc_batch_size:].to(device))*(1-ppo_epsilon_gene) \
                                                            - (1-amdail_alpha)*ppo_mix_confid_batch[disc_batch_size:].to(device)) / (amdail_alpha*(1-ppo_mix_confid_batch[disc_batch_size:].to(device))),              \
                                                       ((amdail_alpha*(1-ppo_mix_confid_batch[disc_batch_size:].to(device)) + (1-amdail_alpha)*ppo_mix_confid_batch[disc_batch_size:].to(device))*(1+ppo_epsilon_gene) \
                                                            - (1-amdail_alpha)*ppo_mix_confid_batch[disc_batch_size:].to(device)) / (amdail_alpha*(1-ppo_mix_confid_batch[disc_batch_size:].to(device)))
        if ratio_flag==True:
            ppo_mix_loss_1  = ppo_mix_reward_batch[:].to(device)                * ppo_mix_ratio[:]
            ppo_real_loss_2 = ppo_mix_reward_batch[:disc_batch_size].to(device) * ppo_mix_ratio[:disc_batch_size].clamp(min=1.0-ppo_epsilon, max=1.0+ppo_epsilon)        
            ppo_gene_loss_2 = ppo_mix_reward_batch[disc_batch_size:].to(device) * ppo_mix_ratio[disc_batch_size:].clamp(min=clip_min,        max=clip_max)
        else:
            ppo_mix_loss_1  = ppo_mix_reward_batch[:].unsqueeze(-1).to(device)                * ppo_mix_ratio[:]
            ppo_real_loss_2 = ppo_mix_reward_batch[:disc_batch_size].unsqueeze(-1).to(device) * ppo_mix_ratio[:disc_batch_size].clamp(min=1.0-ppo_epsilon, max=1.0+ppo_epsilon)        
            ppo_gene_loss_2 = ppo_mix_reward_batch[disc_batch_size:].unsqueeze(-1).to(device) * ppo_mix_ratio[disc_batch_size:].clamp(min=clip_min,        max=clip_max)
        #################################################################
        #################################################################
        ratio_temp    = 0.5
        ppo_real_loss = ppo_real_alpha * torch.min(ppo_mix_loss_1[:disc_batch_size], ppo_real_loss_2) * (1-ratio_temp)
        ppo_gene_loss = ppo_gene_alpha * torch.min(ppo_mix_loss_1[disc_batch_size:], ppo_gene_loss_2) * ratio_temp
        del ppo_mix_loss_1, ppo_real_loss_2, ppo_gene_loss_2
        torch.cuda.empty_cache()
        #################################################################
        sft_real_loss_log.append(ppo_real_loss.mean().item())
        ppo_gene_loss_log.append(ppo_gene_loss.mean().item())
        if ratio_flag==True: ppo_gene_loss_list.extend(ppo_gene_loss.tolist())
        else               : ppo_gene_loss_list.extend(ppo_gene_loss.mean(dim=-1).tolist())
        
        if reward_type=="zero": gene_loss = ppo_real_loss.mean()
        else                  : gene_loss = ppo_real_loss.mean() + ppo_gene_loss.mean()
        del ppo_real_loss, ppo_gene_loss
        torch.cuda.empty_cache()
        #################################################################
        ratio_clipped = ppo_mix_ratio[disc_batch_size:].clamp(min=clip_min, max=clip_max)
        if ratio_flag==True:
            ppo_gene_ratio_log.extend(ppo_mix_ratio[disc_batch_size:].tolist())
            ppo_gene_confid_log.extend(ppo_mix_confid_batch[disc_batch_size:].tolist())
            ppo_gene_ratio_clipped_log.extend(torch.where(ppo_mix_reward_batch[disc_batch_size:].to(device)>0,
                                                          torch.min(ratio_clipped[:], ppo_mix_ratio[disc_batch_size:]),
                                                          torch.max(ratio_clipped[:], ppo_mix_ratio[disc_batch_size:])).tolist())
        else:
            for itr in range(disc_batch_size):
                ppo_gene_ratio_log.extend(ppo_mix_ratio[itr+disc_batch_size].tolist())
                for _ in range(len(ratio_clipped[itr])): ppo_gene_confid_log.append(ppo_mix_confid_batch[itr+disc_batch_size].item())
                if ppo_mix_reward_batch[itr+disc_batch_size]>0: ppo_gene_ratio_clipped_log.extend(torch.min(ratio_clipped[itr], ppo_mix_ratio[itr+disc_batch_size]).tolist())
                else                                          : ppo_gene_ratio_clipped_log.extend(torch.max(ratio_clipped[itr], ppo_mix_ratio[itr+disc_batch_size]).tolist())    
        del ppo_mix_reward_batch, ppo_mix_ratio, ppo_mix_confid_batch
        #################################################################
        gene_loss_log.append(gene_loss.item())
        #################################################################
        gene_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(gene_model.parameters(), max_gradient_norm)
        gene_optimizer.step()
        gene_scheduler.step()
        gene_optimizer.zero_grad()
        del gene_loss
        torch.cuda.empty_cache()
        #################################################################
    if replay_itr%valid_ce_interval==0: utils.functions.hist_gene_loss(seed=seed,
                                                                       dir_path=dir_path,
                                                                       dataset_type=dataset_type,
                                                                       model_type=model_type,
                                                                       model_idx=gene_model_idx,
                                                                       reward_type=reward_type,
                                                                       clip_type=clip_type,
                                                                       ppo_gene_loss=ppo_gene_loss_list)

##################################################################################################################################
# Plot
##################################################################################################################################
print("{}/{} replay_num: {:.2f} mins.".format(replay_itr+1, replay_num, (time.time()-init_time)/60))
utils.functions.plot_ratio(seed=seed,
                           dir_path=dir_path,
                           dataset_type=dataset_type,
                           model_type=model_type,
                           model_idx=gene_model_idx,
                           reward_type=reward_type,
                           clip_type=clip_type,
                           ppo_gene_ratio_log=ppo_gene_ratio_log[-10*buffer_num:],
                           ppo_gene_ratio_clipped_log=ppo_gene_ratio_clipped_log[-10*buffer_num:],
                           ppo_gene_confid_log=ppo_gene_confid_log[-10*buffer_num:])
utils.functions.plot_result(replay_itr=replay_itr,
                            gene_step_num=gene_step_num,
                            valid_ce_interval=valid_ce_interval,
                            seed=seed,
                            dir_path=dir_path,
                            dataset_type=dataset_type,
                            model_type=model_type,
                            gene_model_idx=gene_model_idx,
                            disc_model_idx=disc_model_idx,
                            method_type=method_type,
                            reward_type=reward_type,
                            clip_type=clip_type,
                            ppo_bleu_score_log=ppo_bleu_score_log,
                            ppo_distinct_score_log=ppo_distinct_score_log,
                            confid_gene_upperQ_log=confid_gene_upperQ_log,
                            confid_gene_lowerQ_log=confid_gene_lowerQ_log,
                            confid_gene_max_log=confid_gene_max_log,
                            confid_gene_min_log=confid_gene_min_log,
                            confid_gene_median_log=confid_gene_median_log,
                            ppo_gene_loss_log=ppo_gene_loss_log,
                            sft_real_loss_log=sft_real_loss_log,
                            kld_gene_loss_log=kld_gene_loss_log)

#################################################################
# Save Model
#################################################################
if gene_model_idx>=0:
    torch.save(gene_model.state_dict(), model_path + dataset_type + "/model/" + reward_type + "_" + clip_type + "/gene_model_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.pth")
    torch.save({"ppo_gene_loss_log": ppo_gene_loss_log,
                "disc_loss_log"    : disc_loss_log},
                dir_path + dataset_type + "/data/" + reward_type + "_" + clip_type + "/loss_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.pth")
    torch.save({"ppo_bleu_score_log"    : ppo_bleu_score_log,
                "ppo_distinct_score_log": ppo_distinct_score_log},
                dir_path + dataset_type + "/data/" + reward_type + "_" + clip_type + "/score_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.pth")
    torch.save({"confid_gene_mean"  : confid_gene_mean_log,
                "confid_gene_median": confid_gene_median_log,
                "confid_gene_upperQ": confid_gene_upperQ_log,
                "confid_gene_lowerQ": confid_gene_lowerQ_log,
                "confid_gene_max"   : confid_gene_max_log,
                "confid_gene_min"   : confid_gene_min_log},
                dir_path + dataset_type + "/data/" + reward_type + "_" + clip_type + "/confid_gene_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.pth")
else:
    torch.save(gene_model.state_dict(), model_path + dataset_type + "/model/" + reward_type + "_" + clip_type + "/gene_model_" + str(model_type) + "_" + str(seed) + "seed.pth")
    torch.save({"ppo_gene_loss_log": ppo_gene_loss_log,
                "disc_loss_log"    : disc_loss_log},
                dir_path + dataset_type + "/data/" + reward_type + "_" + clip_type + "/loss_" + str(model_type) + "_" + str(seed) + "seed.pth")
    torch.save({"ppo_bleu_score_log"    : ppo_bleu_score_log,
                "ppo_distinct_score_log": ppo_distinct_score_log},
                dir_path + dataset_type + "/data/" + reward_type + "_" + clip_type + "/score_" + str(model_type) + "_" + str(seed) + "seed.pth")
    torch.save({"confid_gene_mean"  : confid_gene_mean_log,
                "confid_gene_median": confid_gene_median_log,
                "confid_gene_upperQ": confid_gene_upperQ_log,
                "confid_gene_lowerQ": confid_gene_lowerQ_log,
                "confid_gene_max"   : confid_gene_max_log,
                "confid_gene_min"   : confid_gene_min_log},
                dir_path + dataset_type + "/data/" + reward_type + "_" + clip_type + "/confid_gene_" + str(model_type) + "_" + str(seed) + "seed.pth")

print("####################################################################################################################################################")
print("ALL WORK Finished: {:.2f} mins.".format((time.time()-init_time)/60))
print("####################################################################################################################################################")
