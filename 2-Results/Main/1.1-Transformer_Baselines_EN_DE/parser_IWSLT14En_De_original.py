



# %% Imports and Setup


#####-------------------------------- NOTE PARSER IWSLT14 EN-DE NOTE ------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################# IWSLT14 EN-DE ##############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE PARSER IWSLT14 EN-DE NOTE ------------------------------------------------#####



# 📄 parser_IWSLT14En_De.py
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


def get_parser():


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ ============================= IWSLT13En-Fr Training Hyperparameters ==========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser = argparse.ArgumentParser(description='IWSLT14 EN-DE Experiment Config')


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Naming configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--mode_name', default="Seed1_1_EXP3", type=str)

    exp_parser.add_argument('--dataset_name', default="IWSLT14_En_De", type=str,
        help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Training configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅🔵 === Seeds ===🔵✅
    exp_parser.add_argument('--seed1', type=int, default=1, help='global seed 1')
    exp_parser.add_argument('--seed2', type=int, default=1, help='global seed 1')    

    # -------------------------------------------------------------------------------------------------
    # ✅🔵 === Training Configuration ===🔵✅
    exp_parser.add_argument('--epochs', default=50, type=int)              # 🎀📣 50   
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
    # ✅ === DataLoader Configuration ===
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
    # ✅ === Dataset Processing Configuration ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--run_dataset_tokenization', type=bool, default=False,
        help='If True → train SentencePiece tokenizer, encode corpus, and create train/test splits.')

    exp_parser.add_argument('--run_fairseq_preprocessing', type=bool, default=False,
        help='If True → run fairseq-preprocess to build Fairseq binary dataset files.')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Optimizer | Activation Function ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--optimizer1', default="Adam", type=str)
    exp_parser.add_argument(
        '--act_name', default="gelu", type=str,
        help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate, geglu)")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Regularization | Augmentations === 📣 📣 
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔵 === Compatibility for augmentation splits (JSD etc.) ===
    exp_parser.add_argument('--aug-splits', type=int, default=0, help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Model Selection ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # exp_parser.add_argument('--net', default="Transformer", type=str)
    exp_parser.add_argument('--net', default="RTM_Former", type=str, 
                            choices=["Transformer", "RTM_Former" ])
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === RTM_Former Configuration Parameters ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--rtm_alpha', default=0.05, type=float, help='RTM rotational modulation strength')                                  # 🎀 was 0.05| 📣 0.05
    exp_parser.add_argument('--rtm_transition_amplification', default=2.5, type=float, help='RTM transition-conditioned amplification factor')   # 🎀 was 0.5 | 📣 0.5
    exp_parser.add_argument('--rtm_num_layers', default=3, type=int, help='Number of decoder layers enhanced with RTM')                          # 🎀 was 3
    exp_parser.add_argument('--rtm_insertion_type', default='ffn_depth_weighted', type=str, choices=['ffn', 'ffn_depth_weighted', 'cross_attn', 'aurtm', 'sartm'], help='RTM insertion location')        # 🎀📣 Target EN: 'ffn' | Target FR: 'cross_attn'
    #--------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--rtm_start_epoch', default=5, type=int, help='Epoch to start RTM cosine curriculum activation')                    # 🎀📣 5
    exp_parser.add_argument('--rtm_full_epoch', default=15, type=int, help='Epoch where RTM reaches full activation')                            # 🎀📣 15
    exp_parser.add_argument('--rtm_scale', default=0.0, type=float, help='Initial RTM activation scale')
    #--------------------------------------------------------------------------------------------------
    exp_parser.add_argument('--rtm_layer_scale_direction', default='reverse', type=str, choices=['forward', 'reverse', 'reverse_final_fixed'], help='Direction of depth-weighted RTM scaling')
    exp_parser.add_argument('--rtm_layer_scale_min', default=0.70, type=float, help='Minimum RTM depth scale for lowest RTM-enhanced decoder layer')
    exp_parser.add_argument('--rtm_final_layer_scale', default=0.5, type=float, help='Fixed RTM scale for final enhanced decoder layer')    
    #-------------------------------------------------------------------------------------------------- 
    exp_parser.add_argument('--rtm_gate_type', default='transition_contrast', type=str, choices=['transition_magnitude', 'transition_contrast'], help='RTM gate type') 

    #-----------------------------📣 Ablation Studies 📣----------------------------------------------
    exp_parser.add_argument('--rtm_depth_weight', default=True, type=bool, help='Enable RTM depth-weighted layer scaling')
    exp_parser.add_argument('--rtm_rotation', default=True, type=bool, help='Enable RTM rotational cos/sin refinement') 
    exp_parser.add_argument('--rtm_register_module', default=False, type=bool, help='Register RTM as decoder module for architecture visibility and parameter tracking')   
    exp_parser.add_argument('--rtm_register_module1', default=True, type=bool, help='Register RTM as decoder module for architecture visibility and parameter tracking')  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Generation / Beam Search Parameters ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--beam', default=5, type=int, help='Beam size for generation (5)')    
    exp_parser.add_argument('--lenpen', default=1.0, type=float, help='Length penalty for beam search (1.0)') 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Report ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    exp_parser.add_argument('--decimal_places', default="four", type=str,
                            choices=["four", "two"], help='Decimal precision for model summary reporting')                            
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    return exp_parser 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────







# %%
