import random
import copy
def shuffle_buffer(text_buffer, reward_buffer=None, confid_buffer=None):
    buffer_num = len(text_buffer)
    buffer_idx = random.sample(range(buffer_num), k=buffer_num)
    
    text_buffer_ = copy.deepcopy(text_buffer)
    for itr in range(buffer_num): text_buffer_[itr] = text_buffer[buffer_idx[itr]]
    if reward_buffer is not None:
        reward_buffer_ = copy.deepcopy(reward_buffer)
        for itr in range(buffer_num): reward_buffer_[itr] = reward_buffer[buffer_idx[itr]]
    if confid_buffer is not None:
        confid_buffer_ = copy.deepcopy(confid_buffer)
        for itr in range(buffer_num): confid_buffer_[itr] = confid_buffer[buffer_idx[itr]]

    if reward_buffer is not None and confid_buffer is not None: return text_buffer_, reward_buffer_, confid_buffer_
    elif reward_buffer is not None: return text_buffer_, reward_buffer_
    else: return text_buffer_
def shuffle_pair(gene_text_buffer, real_text_buffer):
    buffer_num = len(gene_text_buffer)
    buffer_idx = random.sample(range(buffer_num), k=buffer_num)
    
    gene_text_buffer_ = copy.deepcopy(gene_text_buffer)
    real_text_buffer_ = copy.deepcopy(real_text_buffer)
    for itr in range(buffer_num):
        gene_text_buffer_[itr] = gene_text_buffer[buffer_idx[itr]]
        real_text_buffer_[itr] = real_text_buffer[buffer_idx[itr]]
    return gene_text_buffer_, real_text_buffer_

import collections
import nltk
nltk.download("punkt")
def compute_distinct(input, n=2):
    counter = collections.Counter()
    total_count = 0
    for item in input:
        hyp = nltk.word_tokenize(item.lower())
        ngrams_list = list(nltk.ngrams(hyp, n=n))
        counter.update(ngrams_list)
        total_count += len(ngrams_list)
    return len(counter) / max(total_count, 1)

import matplotlib.pyplot as plt
import numpy as np
def plot_result(dir_path,
                seed,
                replay_itr,
                gene_step_num,
                dataset_type,
                method_type,
                model_type,
                gene_model_idx=-1,
                disc_model_idx=-1,
                reward_type=None,
                clip_type=None,
                train_ce_loss_log=None,
                valid_ce_loss_log=None,
                valid_ce_interval=None,
                ppo_bleu_score_log=None,
                ppo_distinct_score_log=None,
                confid_gene_upperQ_log=None,
                confid_gene_lowerQ_log=None,
                confid_gene_max_log=None,
                confid_gene_min_log=None,
                confid_gene_median_log=None,
                ppo_gene_loss_log=None,
                sft_real_loss_log=None,
                kld_gene_loss_log=None):
    plt.tight_layout()
    plt.rcParams["font.size"] = 14
    if method_type in ["sft"]:
        valid_x = np.linspace(start=0, stop=replay_itr//valid_ce_interval*valid_ce_interval*gene_step_num, num=replay_itr//valid_ce_interval+1)
        if method_type=="sft":
            plt.plot(np.log(train_ce_loss_log),          alpha=0.5, marker=".", markersize=1, label="CE Train", linestyle="None")
            plt.plot(valid_x, np.log(valid_ce_loss_log), alpha=1.0, marker=".", markersize=1, label="CE Valid")
        plt.grid()
        plt.legend()
        if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/loss_sft1_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx.png",
                                          bbox_inches='tight', pad_inches=0.1)
        else                : plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/loss_sft1_" + str(model_type) + "_" + str(seed) + "seed.png",
                                          bbox_inches='tight', pad_inches=0.1)
        #################################################################
        plt.ylim(0.75, 1.25)
        if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/loss_sft2_" + str(model_type) + "_" + str(seed) + "seed_" + str(gene_model_idx) + "idx.png",
                                          bbox_inches='tight', pad_inches=0.1)
        else                : plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/loss_sft2_" + str(model_type) + "_" + str(seed) + "seed.png",
                                          bbox_inches='tight', pad_inches=0.1)
        plt.close()
        #################################################################
    else:
        plt.plot(np.array(confid_gene_upperQ_log), color="red",   alpha=0.5, linestyle="None", marker=".", markersize=2)
        plt.plot(np.array(confid_gene_lowerQ_log), color="red",   alpha=0.5, linestyle="None", marker=".", markersize=2)
        plt.plot(np.array(confid_gene_max_log),    color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
        plt.plot(np.array(confid_gene_min_log),    color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
        plt.plot(np.array(confid_gene_median_log), color="blue",  alpha=0.5, linestyle="None", marker=".", markersize=2)
        plt.ylim(0.0, 1.0)
        plt.grid()
        if method_type=="disc":
            if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/confid_gene_" + str(model_type) + "_"  + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.png",
                                              bbox_inches='tight', pad_inches=0.1)
            else                : plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/confid_gene_" + str(model_type) + "_"  + str(seed) + "seed.png",
                                              bbox_inches='tight', pad_inches=0.1)
        else:
            if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/confid_gene_" + str(model_type) + "_"  + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.png",
                                              bbox_inches='tight', pad_inches=0.1)
            else                : plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/confid_gene_" + str(model_type) + "_"  + str(seed) + "seed.png",
                                              bbox_inches='tight', pad_inches=0.1)
        plt.close()
    #################################################################
    if method_type not in ["sft", "disc"]:
        plt.scatter(np.array(ppo_bleu_score_log), np.array(ppo_distinct_score_log),
                    color="black", alpha=0.5, marker=".")
        if replay_itr!=0:
            plt.xlim(0.10, 0.14)
            plt.ylim(0.65, 0.85)
        plt.grid()
        if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/tradeoff_" + str(model_type) + "_"  + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.png",
                                          bbox_inches='tight', pad_inches=0.1)
        else                : plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/tradeoff_" + str(model_type) + "_"  + str(seed) + "seed.png",
                                          bbox_inches='tight', pad_inches=0.1)
        plt.close()
                
        plt.plot(sft_real_loss_log, alpha=0.5, linestyle="None", marker=".", markersize=2, label="SFT")
        plt.plot(ppo_gene_loss_log, alpha=0.5, linestyle="None", marker=".", markersize=2, label="PPO")
        plt.grid()
        plt.legend()
        if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/loss_poli_" + str(model_type) + "_"  + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.png",
                                          bbox_inches='tight', pad_inches=0.1)
        else                : plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/loss_poli_" + str(model_type) + "_"  + str(seed) + "seed.png",
                                          bbox_inches='tight', pad_inches=0.1)
        plt.close()

def plot_bound(seed,
               dir_path,
               dataset_type,
               method_type,
               real_UB_mean_log=None,
               real_UB_median_log=None,
               real_UB_upperQ_log=None,
               real_UB_lowerQ_log=None,
               real_UB_max_log=None,
               real_UB_min_log=None,
               gene_UB_mean_log=None,
               gene_UB_median_log=None,
               gene_UB_upperQ_log=None,
               gene_UB_lowerQ_log=None,
               gene_UB_max_log=None,
               gene_UB_min_log=None,
               real_LB_mean_log=None,
               real_LB_median_log=None,
               real_LB_upperQ_log=None,
               real_LB_lowerQ_log=None,
               real_LB_max_log=None,
               real_LB_min_log=None,
               gene_LB_mean_log=None,
               gene_LB_median_log=None,
               gene_LB_upperQ_log=None,
               gene_LB_lowerQ_log=None,
               gene_LB_max_log=None,
               gene_LB_min_log=None):
    plt.tight_layout()
    plt.rcParams["font.size"] = 14
    plt.plot(np.array(real_UB_upperQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_UB_lowerQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_UB_max_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_UB_min_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_UB_median_log), color="blue", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_UB_mean_log), color="green", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.grid()
    plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/UB_real_" + str(seed) + "seed.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    plt.plot(np.array(gene_UB_upperQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_UB_lowerQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_UB_max_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_UB_min_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_UB_median_log), color="blue", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_UB_mean_log), color="green", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.grid()
    plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/UB_gene_" + str(seed) + "seed.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.plot(np.array(real_LB_upperQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_LB_lowerQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_LB_max_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_LB_min_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_LB_median_log), color="blue", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(real_LB_mean_log), color="green", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.grid()
    plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/LB_real_" + str(seed) + "seed.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    plt.plot(np.array(gene_LB_upperQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_LB_lowerQ_log), color="red", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_LB_max_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_LB_min_log), color="black", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_LB_median_log), color="blue", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.plot(np.array(gene_LB_mean_log), color="green", alpha=0.5, linestyle="None", marker=".", markersize=2)
    plt.grid()
    plt.savefig(dir_path + dataset_type + "/figure/" + method_type + "/LB_gene_" + str(seed) + "seed.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

import matplotlib.cm as cm
import seaborn as sns
from matplotlib.gridspec import GridSpec
def plot_ratio(seed,
               dir_path,
               dataset_type,
               reward_type,
               clip_type,
               model_type,
               model_idx=-1,
               ppo_real_ratio_log=None,
               ppo_real_ratio_clipped_log=None,
               ppo_real_confid_log=None,
               ppo_gene_ratio_log=None,
               ppo_gene_ratio_clipped_log=None,
               ppo_gene_confid_log=None):
    plt.tight_layout()
    plt.rcParams["font.size"] = 14
    epsilon = 0.2
    
    ppo_gene_ratio_log = np.array(ppo_gene_ratio_log)
    ppo_gene_ratio_clipped_log = np.array(ppo_gene_ratio_clipped_log)
    ppo_gene_confid_log = np.array(ppo_gene_confid_log)
    if ppo_real_ratio_log is None:
        x_max = 2.0
        x_min = 0.0
    else:
        ppo_real_ratio_log = np.array(ppo_real_ratio_log)
        ppo_real_ratio_clipped_log = np.array(ppo_real_ratio_clipped_log)
        ppo_real_confid_log = np.array(ppo_real_confid_log)
        x_max = 2.0
        x_min = 0.0
    #################################################################
    fig = plt.figure()
    grid = GridSpec(12, 12)
    grid.tight_layout(fig)

    ax_scatter = fig.add_subplot(grid[4:, :-4],
                                 xlim=(x_min, x_max),
                                 ylim=(0.0, 1.0))
    ax_hist_x = fig.add_subplot(grid[:3, :-4])
    ax_hist_y = fig.add_subplot(grid[4:, -3:])

    ax_scatter.grid()
    if ppo_real_ratio_log is not None:
        ax_scatter.scatter(x=ppo_real_ratio_clipped_log,
                           y=ppo_real_confid_log,
                           s=1.0,
                           alpha=0.5,
                           color="tab:blue")
        ax_scatter.axvline(x=np.median(ppo_real_ratio_clipped_log),
                           linestyle="--",
                           linewidth=1.0,
                           color="black")
        ax_scatter.axhline(y=np.median(ppo_real_confid_log),
                           linestyle="--",
                           linewidth=1.0,
                           color="black")
        ax_scatter.set_xlabel("PPO Ratio")
        ax_scatter.set_ylabel("Confidence")
        #################################################################
        ax_hist_x.grid()
        ax_hist_x.set_xlim(x_min, x_max)
        ax_hist_x.hist(ppo_real_ratio_clipped_log,
                       bins=32,
                       weights=np.ones(len(ppo_real_ratio_clipped_log))/len(ppo_real_ratio_clipped_log))
        ax_hist_x.axvline(x=np.median(ppo_real_ratio_clipped_log-1.0),
                          linestyle="--",
                          linewidth=1.0,
                          color="black")
        ax_hist_y.grid()
        ax_hist_y.set_ylim(0.0, 1.0)
        ax_hist_y.hist(ppo_real_confid_log,
                       bins=32,
                       weights=np.ones(len(ppo_real_ratio_clipped_log))/len(ppo_real_ratio_clipped_log),
                       orientation="horizontal")
        ax_hist_y.axhline(y=np.median(ppo_real_confid_log),
                          linestyle="--",
                          linewidth=1.0,
                          color="black")
        #################################################################
        if model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/ratio_real_" + str(model_type) + "_"  + str(seed) + "seed_" + str(model_idx) + "idx.png",
                                    bbox_inches='tight', pad_inches=0.1)
        else: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/ratio_real_" + str(model_type) + "_"  + str(seed) + "seed.png",
                                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    #################################################################
    fig = plt.figure()
    grid = GridSpec(12, 12)
    grid.tight_layout(fig)

    ax_scatter = fig.add_subplot(grid[4:, :-4],
                                 xlim=(x_min, x_max),
                                 ylim=(0.0, 1.0))
    ax_hist_x = fig.add_subplot(grid[:3, :-4])
    ax_hist_y = fig.add_subplot(grid[4:, -3:])

    ax_scatter.grid()
    ax_scatter.scatter(x=ppo_gene_ratio_clipped_log,
                       y=ppo_gene_confid_log,
                       s=1.0,
                       alpha=0.5,
                       color="tab:blue")
    ax_scatter.axvline(x=np.median(ppo_gene_ratio_clipped_log),
                       linestyle="--",
                       linewidth=1.0,
                       color="black")
    ax_scatter.axvline(x=1+epsilon,
                       linestyle="--",
                       linewidth=1.0,
                       color="black")
    ax_scatter.axvline(x=1-epsilon,
                       linestyle="--",
                       linewidth=1.0,
                       color="black")
    ax_scatter.axhline(y=np.median(ppo_gene_confid_log),
                       linestyle="--",
                       linewidth=1.0,
                       color="black")
    ax_scatter.set_xlabel("PPO Ratio")
    ax_scatter.set_ylabel("Confidence")
    #################################################################
    ax_hist_x.grid()
    ax_hist_x.set_xlim(x_min, x_max)
    ax_hist_x.hist(ppo_gene_ratio_clipped_log,
                   range=(x_min, x_max),
                   bins=64,
                   weights=np.ones(len(ppo_gene_ratio_clipped_log))/len(ppo_gene_ratio_clipped_log))
    ax_hist_x.axvline(x=np.median(ppo_gene_ratio_clipped_log),
                      linestyle="--",
                      linewidth=1.0,
                      color="black")
    ax_hist_x.axvline(x=1+epsilon,
                      linestyle="--",
                      linewidth=1.0,
                      color="black")
    ax_hist_x.axvline(x=1-epsilon,
                      linestyle="--",
                      linewidth=1.0,
                      color="black")
    #################################################################
    ax_hist_y.grid()
    ax_hist_y.set_ylim(0.0, 1.0)
    ax_hist_y.hist(ppo_gene_confid_log,
                   bins=32,
                   weights=np.ones(len(ppo_gene_ratio_clipped_log))/len(ppo_gene_ratio_clipped_log),
                   orientation="horizontal")
    ax_hist_y.axhline(y=np.median(ppo_gene_confid_log),
                      linestyle="--",
                      linewidth=1.0,
                      color="black")
    #################################################################
    if model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/ratio_gene_" + str(model_type) + "_"  + str(seed) + "seed_" + str(model_idx) + "idx.png",
                                 bbox_inches='tight', pad_inches=0.1)
    else: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/ratio_gene_" + str(model_type) + "_"  + str(seed) + "seed.png",
                      bbox_inches='tight', pad_inches=0.1)
    plt.close()

def hist_reward(seed,
                dir_path,
                dataset_type,
                reward_type,
                clip_type,
                model_type,
                gene_model_idx=-1,
                disc_model_idx=-1,
                gene_reward_buffer=None):
    plt.tight_layout()
    plt.rcParams["font.size"] = 14
    plt.hist(gene_reward_buffer,
             bins=32,
             weights=np.ones(len(gene_reward_buffer))/len(gene_reward_buffer))
    plt.xlabel("Normalized Reward Value")
    plt.ylabel("Frequency")
    plt.grid()
    if gene_model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/hist_gene_reward_" + str(model_type) + "_"  + str(seed) + "seed_" + str(gene_model_idx) + "idx_" + str(disc_model_idx) + "idx.png",
                                      bbox_inches='tight', pad_inches=0.1)
    else: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/hist_gene_reward_" + str(model_type) + "_"  + str(seed) + "seed.png",
                      bbox_inches='tight', pad_inches=0.1)
    plt.close()

def hist_gene_loss(seed,
                   dir_path,
                   dataset_type,
                   model_type,
                   reward_type,
                   clip_type,
                   model_idx=-1,
                   ppo_gene_loss=None):
    ppo_gene_loss = np.array(ppo_gene_loss)
    plt.tight_layout()
    plt.rcParams["font.size"] = 14
    plt.hist(ppo_gene_loss,
             bins=32,
             weights=np.ones(len(ppo_gene_loss))/len(ppo_gene_loss))
    plt.grid()
    if model_idx>=0: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/hist_gene_loss_" + str(model_type) + "_"  + str(seed) + "seed_" + str(model_idx) + "idx.png",
                                 bbox_inches='tight', pad_inches=0.1)
    else: plt.savefig(dir_path + dataset_type + "/figure/" + reward_type + "_" + clip_type + "/hist_gene_loss_" + str(model_type) + "_"  + str(seed) + "seed.png",
                      bbox_inches='tight', pad_inches=0.1)
    plt.close()
