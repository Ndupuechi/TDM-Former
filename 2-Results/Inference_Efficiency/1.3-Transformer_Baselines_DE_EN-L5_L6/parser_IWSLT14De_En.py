



# %% Imports and Setup


#####-------------------------------- NOTE PARSER IWSLT14 DE-EN NOTE ------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################# IWSLT14 DE-EN ##############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE PARSER IWSLT14 DE-EN NOTE ------------------------------------------------#####



# 📄 parser_IWSLT14De_En.py
########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################

# ======================================================================================================
# 📜 === Core Libraries ===
# ======================================================================================================

import argparse


# 📣 📣 🔵 ❌❌🟦⭐🧪🧪📦📦⚖️⚖️🔖🚀🅰️🔼🅱️🔼🔖✅🎀
########################################################################################################################
####-------| NOTE 2.1. ARGUMENT PARSER | XXX -------------------------------------------------------####################
########################################################################################################################



import argparse


# 📣 📣 🔵 ❌❌🟦⭐🧪🧪📦📦⚖️⚖️🔖🚀🅰️🔼🅱️🔼🔖✅🎀
########################################################################################################################
####-------| NOTE 2.1. ARGUMENT PARSER | XXX -------------------------------------------------------####################
########################################################################################################################


def get_parser():


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 0️⃣ ============================= IWSLT14De-En Training Hyperparameters ==========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser = argparse.ArgumentParser(description='IWSLT14 DE-EN Experiment Config')


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ === Naming configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--mode_name', default="Seed1_1_TDM_NoRotation_Last2Layer", type=str,
        choices=[

            # ────────────────────────────────────────────────────────────────────────
            # 🧪🧪 === Baseline Transformer ===
            # ────────────────────────────────────────────────────────────────────────
            "Seed1_1_Baseline_Transformer",

            # ────────────────────────────────────────────────────────────────────────
            # 🧩🧩 === TDM insertion-depth ablation (last N decoder FFN layers) ===
            # ────────────────────────────────────────────────────────────────────────
            "Seed1_1_TDM_NoRotation_Last1Layer", "Seed1_1_TDM_NoRotation_Last2Layer", "Seed1_1_TDM_NoRotation_Last3Layer", "Seed1_1_TDM_NoRotation_Last4Layer",
            "Seed1_1_TDM_WithRotation_Last1Layer", "Seed1_1_TDM_WithRotation_Last2Layer", "Seed1_1_TDM_WithRotation_Last3Layer", "Seed1_1_TDM_WithRotation_Last4Layer",

            # ────────────────────────────────────────────────────────────────────────
            # 🚦🚦 === TDM component ablation ===
            # ────────────────────────────────────────────────────────────────────────
            "Seed1_1_TDM_NoRotation", "Seed1_1_TDM_WithRotation"

        ],
        help="Choose experiment name for result, log, and checkpoint file naming. "
            "🟦🟦 Baseline_Transformer = Transformer: standard Fairseq Transformer baseline. "
            "🧬🧬 TDM_NoRotation = TDM-Former: transition-conditioned FFN modulation without rotational refinement. "
            "🌀🌀 TDM_WithRotation = TDM-Former-R: TDM-Former with rotational transition refinement, inserted into the selected last decoder FFN layers. "
            "📣📣 Seed1_1_TDM_NoRotation_Last3Layer | Seed1_1_TDM_WithRotation_Last3Layer ➡️  Seed1_1_TDM_NoRotation | Seed1_1_TDM_WithRotation (in Ablation Experiment). "
        )
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--dataset_name', default="IWSLT14_De_En", type=str,
        help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ === Training configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅🔵 === Seeds ===🔵✅
    exp_parser.add_argument('--seed1', type=int, default=1, help='global seed 1')
    exp_parser.add_argument('--seed2', type=int, default=1, help='global seed 1')    

    # -------------------------------------------------------------------------------------------------
    # ✅🔵 === Training Configuration ===🔵✅
    exp_parser.add_argument('--epochs', default=50, type=int)          
    exp_parser.add_argument('--start_epoch', default=0, type=int)    

    # -------------------------------------------------------------------------------------------------
    # ✅🔵 === Learning rate schedule ===🔵✅
    exp_parser.add_argument('--warmup_updates', default=4000, type=int,
        help="Number of warmup steps before starting inverse square root decay")
    exp_parser.add_argument('--lr', default=1e-4, type=float)  

    # -------------------------------------------------------------------------------------------------
    # ✅🔵 === Weights and Smoothing ===🔵✅
    exp_parser.add_argument('--weight_decay', default=1e-4, type=float)
    exp_parser.add_argument('--eps', default=1e-9, type=float)
    exp_parser.add_argument('--smoothing', default=0.1, type=float)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 3️⃣ === DataLoader Configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅🔵 === Data loader ===🔵✅
    exp_parser.add_argument('--max_tokens', default=4096, type=int,     
        help="Maximum number of tokens per batch (default: 4096, paper standard for IWSLT14)")
    exp_parser.add_argument('--max_source_positions', default=256, type=int,   
        help="Maximum source sequence length (default: 1024)")
    exp_parser.add_argument('--max_target_positions', default=256, type=int, 
        help="Maximum target sequence length (default: 1024)")
    exp_parser.add_argument('--num_workers', default=0, type=int,    
        help="Number of dataloader workers (default: 4)")
    exp_parser.add_argument('--num_shards', default=1, type=int,
        help="Number of dataset shards (default: 1 = no sharding)")
    exp_parser.add_argument('--shard_id', default=0, type=int,
        help="ID of this shard (default: 0)")
    exp_parser.add_argument('--ignore_invalid_inputs', type=bool, default=True,
        help="Ignore invalid input samples instead of crashing (default: True)")
    # -------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--vocab_size', default=10000, type=int,
        help="SentencePiece vocabulary size")    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣ === Dataset Processing Configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--run_dataset_tokenization', type=bool, default=False,
        help='If True → train SentencePiece tokenizer, encode corpus, and create train/test splits.')

    exp_parser.add_argument('--run_fairseq_preprocessing', type=bool, default=False,
        help='If True → run fairseq-preprocess to build Fairseq binary dataset files.')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 5️⃣ === Optimizer | Activation Function ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--optimizer1', default="Adam", type=str)
    exp_parser.add_argument(
        '--act_name', default="gelu", type=str,
        help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate, geglu)")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 6️⃣ === Regularization | Augmentations === 📣 📣 
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔵 === Compatibility for augmentation splits (JSD etc.) ===
    exp_parser.add_argument('--aug-splits', type=int, default=0, help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 7️⃣ === Model Selection ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # exp_parser.add_argument('--net', default="Transformer", type=str)
    exp_parser.add_argument('--net', default="TDM_Former", type=str, 
                            choices=["Transformer", "TDM_Former" ])
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 8️⃣ === TDM_Former Configuration Parameters ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--tdm_alpha', default=0.05, type=float, help='TDM rotational modulation strength')                                                    # 🎀📣 0.05
    exp_parser.add_argument('--tdm_transition_amplification', default=4.5, type=float, help='TDM transition-conditioned amplification factor')                     # 🎀📣 4.5
    exp_parser.add_argument('--tdm_num_layers', default=2, type=int, help='Number of decoder layers enhanced with TDM')                                            # 🎀📣 3
    exp_parser.add_argument('--tdm_insertion_type', default='ffn', type=str, choices=['ffn'], help='TDM insertion location')                                       # 🎀📣 ffn
    #--------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--tdm_start_epoch', default=5, type=int, help='Epoch to start TDM cosine curriculum activation')                                      # 🎀📣 5
    exp_parser.add_argument('--tdm_full_epoch', default=15, type=int, help='Epoch where TDM reaches full activation')                                              # 🎀📣 15
    exp_parser.add_argument('--tdm_scale', default=0.0, type=float, help='Initial TDM activation scale')
    exp_parser.add_argument('--tdm_gate_type', default='transition_contrast', type=str, choices=['transition_magnitude', 'transition_contrast'], help='TDM gate type')
    #--------------------------------------------------------------------------------------------------
    

    #--------------------------------------------------------------------------------------------------
    #------------------🧪 Ablation Studies: TDM Component Ablation Settings 🧪------------------------
    #----------------🧬 Main TDM Mechanism + 🌀 Optional Rotation Extension 🧪-----------------------
    #--------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--tdm_transition', default=True, type=bool, help='🧬 Enable core transition dynamics modulation | 🔑🏷️ TDM_NoRotation ➡️ TDM-Former')                                     
    exp_parser.add_argument('--tdm_rotation', default=False, type=bool, help='🌀 Enable optional rotational transition refinement | ✨🏷️ TDM_WithRotation ➡️ TDM-Former-R')    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 9️⃣ === Generation / Beam Search Parameters ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--beam', default=5, type=int, help='Beam size for generation (5)')    
    exp_parser.add_argument('--lenpen', default=1.0, type=float, help='Length penalty for beam search (1.0)') 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣0️⃣ === Report ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--decimal_places', default="four", type=str,
                            choices=["four", "two"], help='Decimal precision for model summary reporting')  
    #--------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--tdm_register_module', default=False, type=bool, help='Register TDM as decoder module for architecture visibility and parameter tracking')   
    exp_parser.add_argument('--tdm_register_module1', default=True, type=bool, help='Register TDM as decoder module for architecture visibility and parameter tracking')  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣1️⃣ === Checkpoint Testing Configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--crash_run', default=True, type=bool, help='False: reset test result file | True: continue after crash and append results')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    return exp_parser 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────







# %%
