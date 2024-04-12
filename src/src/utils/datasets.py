import json
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Seq2SeqDataset(Dataset):
    """
    A Simple Seq2Seq Dataset Implementation
    """
    def __init__(self, dataset, gene_tokenizer, disc_tokenizer, dataset_type):
        self.dataset = dataset
        self.gene_tokenizer = gene_tokenizer
        self.disc_tokenizer = disc_tokenizer
        self.dataset_type = dataset_type
    
    def __getitem__(self, index):
        item = self.dataset[index]
        #################################################################
        if self.dataset_type=="common_gen":
            item["concept_set_idx"] = torch.LongTensor([item["concept_set_idx"]])
            source_text = item["concepts"]
            source_text = " ".join(source_text)
            target_text = item["target"]
        elif self.dataset_type=="roc_story":
            sentence_1 = item["sentence1"]
            sentence_2 = item["sentence2"]
            sentence_3 = item["sentence3"]
            sentence_4 = item["sentence4"]

            source_text = sentence_1 + " " + sentence_2 + " " + sentence_3 + " " + sentence_4
            target_text = item["sentence5"]
        elif self.dataset_type=="xsum":
            source_text = item["document"]
            target_text = item["summary"]
        elif self.dataset_type=="cnn_dailymail":
            source_text = item["article"]
            target_text = item["highlights"]
        elif self.dataset_type=="gigaword":
            source_text = item["document"]
            target_text = item["summary"]
        elif self.dataset_type=="samsum":
            source_text = item["dialogue"]
            target_text = item["summary"]
        else:
            print("dataset_type is false.")
            exit()
        #################################################################
        source_text = re.split(" +", source_text)
        source_text = " ".join(source_text)
        source_text = source_text.strip()

        target_text = re.split(" +", target_text)
        target_text = " ".join(target_text)
        target_text = target_text.strip()
        
        g_source_text = self.gene_tokenizer.bos_token + " " + source_text
        g_target_text = " " + self.gene_tokenizer.sep_token + " " + target_text + " " + self.gene_tokenizer.eos_token
        g_joint_text  = g_source_text + g_target_text
        
        d_source_text = self.disc_tokenizer.bos_token + " " + source_text
        d_target_text = " " + self.disc_tokenizer.sep_token + " " + target_text + " " + self.disc_tokenizer.eos_token
        d_joint_text  = d_source_text + d_target_text
        #################################################################
        g_source_input_ids = self.gene_tokenizer(g_source_text, add_special_tokens=False)["input_ids"]
        g_target_input_ids = self.gene_tokenizer(g_target_text, add_special_tokens=False)["input_ids"]
        g_joint_input_ids  = self.gene_tokenizer(g_joint_text,  add_special_tokens=False)["input_ids"]

        d_source_input_ids = self.disc_tokenizer(d_source_text, add_special_tokens=False)["input_ids"]
        d_target_input_ids = self.disc_tokenizer(d_target_text, add_special_tokens=False)["input_ids"]
        d_joint_input_ids  = self.disc_tokenizer(d_joint_text,  add_special_tokens=False)["input_ids"]
        #################################################################
        item["g_source_input_ids"] = torch.LongTensor(g_source_input_ids)
        item["g_target_input_ids"] = torch.LongTensor(g_target_input_ids)
        item["g_joint_input_ids"]  = torch.LongTensor(g_joint_input_ids)

        item["d_source_input_ids"] = torch.LongTensor(d_source_input_ids)
        item["d_target_input_ids"] = torch.LongTensor(d_target_input_ids)
        item["d_joint_input_ids"]  = torch.LongTensor(d_joint_input_ids)
        #################################################################
        if self.dataset_type=="common_gen" : del item["concepts"], item["target"]
        elif self.dataset_type=="roc_story": del item["sentence1"], item["sentence2"], item["sentence3"], item["sentence4"], item["sentence5"]
        torch.cuda.empty_cache()
        return item
    
    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        new_batch = {}
        if self.dataset_type=="common_gen": new_batch["concept_set_idx"] = torch.LongTensor([item["concept_set_idx"].item() for item in batch])
        
        #################################################################
        # Generator
        #################################################################
        new_batch["g_source_input_ids_padded"] = pad_sequence([item["g_source_input_ids"] for item in batch],
                                                              batch_first=True,
                                                              padding_value=self.gene_tokenizer.pad_token_id)
        new_batch["g_target_input_ids_padded"] = pad_sequence([item["g_target_input_ids"] for item in batch],
                                                              batch_first=True,
                                                              padding_value=self.gene_tokenizer.pad_token_id)
        new_batch["g_joint_input_ids_padded"]  = torch.cat((new_batch["g_source_input_ids_padded"], \
                                                            new_batch["g_target_input_ids_padded"]), dim=1)
        
        #################################################################
        # Discriminator
        #################################################################
        new_batch["d_source_input_ids_padded"] = pad_sequence([item["d_source_input_ids"] for item in batch],
                                                              batch_first=True,
                                                              padding_value=self.disc_tokenizer.pad_token_id)
        new_batch["d_target_input_ids_padded"] = pad_sequence([item["d_target_input_ids"] for item in batch],
                                                              batch_first=True,
                                                              padding_value=self.disc_tokenizer.pad_token_id)
        new_batch["d_joint_input_ids_padded"]  = torch.cat((new_batch["d_source_input_ids_padded"], \
                                                            new_batch["d_target_input_ids_padded"]), dim=1)

        #################################################################
        # No Padding
        #################################################################
        new_batch["g_source_input_ids"] = [item["g_source_input_ids"] for item in batch]
        new_batch["g_joint_input_ids"]  = [item["g_joint_input_ids"] for item in batch]
        new_batch["d_source_input_ids"] = [item["d_source_input_ids"] for item in batch]
        new_batch["d_joint_input_ids"]  = [item["d_joint_input_ids"] for item in batch]
        
        del batch
        torch.cuda.empty_cache()
        
        return new_batch
