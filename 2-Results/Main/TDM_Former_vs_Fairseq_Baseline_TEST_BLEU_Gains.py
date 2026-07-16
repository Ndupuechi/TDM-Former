



# %% Imports and Setup


# ===============================================================
# 🔗==================== ANALYSIS 🔑==========================🔗
# 🔗==============⚖️ IWSLT DATASET ===========================🔗
# ===============================================================


########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################


# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Alternative for memory split limits:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ======================================================================================================
# ✅ === Standard libraries ===
# ======================================================================================================
import re
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.lines import Line2D
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 2. DEFINE DIRECTORY PATH | XXX ---------------------------------------------------####################
########################################################################################################################

# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Transformers\Github\2-Results\Main"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ✅ === Define core project paths ===
PROJECT_PATH = Project_PATH
MODELS_PATH = os.path.join(Project_PATH, "models")
ACTIVATION_PATH = os.path.join(Project_PATH, "activation")


# ✅ === Add essential paths to sys.path ===
for path in [PROJECT_PATH, MODELS_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 3. DEFINE PARSAR DATA| XXX -----------------------------------------------------######################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 0️⃣ === Dataset information === 
exp_parser = argparse.ArgumentParser("IWSLT Experiment Config")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 === Evaluation mode ===
exp_parser.add_argument('--evaluation_mode', default="test", type=str, help="Choose dataset: [test, train]")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === Define curves : GENERAL 📣 📣 ==========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === Define curves : GENERAL 📣 📣 ==========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ === Training configuration ===
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
# 2️⃣ === DataLoader Configuration ===
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
# 3️⃣ === Optimizer | Activation Function ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
exp_parser.add_argument('--optimizer1', default="Adam", type=str)
exp_parser.add_argument(
    '--act_name', default="gelu", type=str,
    help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate, geglu)")
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣⚙️ === Define curves : SPECIFIC 📣 📣 =======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣⚙️ === Define curves : SPECIFIC 📣 📣 =======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ====================================================================================================================
# ===================== 4️⃣.1️⃣ 📌📌 ENGLISH ⬅️➡️ GERMAN  =========================================================== 
# ====================================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.1️⃣.1️⃣🅰️ Special mode curve: PLOT1_1 🔖🔖 | English→German 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot1_1A', default="PLOT1_TDM_Former_En_De", type=str)
exp_parser.add_argument('--mode_name_Plot1_1A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot1_1A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot1_1A', default="IWSLT14_En_De", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.1️⃣.1️⃣🅱️ Special mode curve: PLOT1_1 🔖🔖 | English→German 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot1_1B', default="PLOT1_Baseline_En_De", type=str)
exp_parser.add_argument('--mode_name_Plot1_1B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot1_1B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot1_1B', default="IWSLT14_En_De", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.1️⃣.2️⃣🅰️ Special mode curve: PLOT1_2 🔖🔖 | German→English 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot1_2A', default="PLOT1_TDM_Former_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot1_2A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot1_2A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot1_2A', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.1️⃣.2️⃣🅱️ Special mode curve: PLOT1_2 🔖🔖 | German→English 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot1_2B', default="PLOT1_Baseline_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot1_2B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot1_2B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot1_2B', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ====================================================================================================================
# ===================== 4️⃣.2️⃣ 📌📌 ENGLISH ⬅️➡️ ROMANIAN  ========================================================= 
# ====================================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.2️⃣.1️⃣🅰️ Special mode curve: PLOT2_1 🔖🔖 | English→Romanian 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot2_1A', default="PLOT2_TDM_Former_En_Ro", type=str)
exp_parser.add_argument('--mode_name_Plot2_1A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot2_1A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot2_1A', default="IWSLT17_En_Ro", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Ro' = English→Romanian, 'IWSLT17_Ro_En' = Romanian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.2️⃣.1️⃣🅱️ Special mode curve: PLOT2_1 🔖🔖 | English→Romanian 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot2_1B', default="PLOT2_Baseline_En_Ro", type=str)
exp_parser.add_argument('--mode_name_Plot2_1B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot2_1B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot2_1B', default="IWSLT17_En_Ro", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Ro' = English→Romanian, 'IWSLT17_Ro_En' = Romanian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.2️⃣.2️⃣🅰️ Special mode curve: PLOT2_2 🔖🔖 | Romanian→English 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot2_2A', default="PLOT2_TDM_Former_Ro_En", type=str)
exp_parser.add_argument('--mode_name_Plot2_2A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot2_2A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot2_2A', default="IWSLT17_Ro_En", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Ro' = English→Romanian, 'IWSLT17_Ro_En' = Romanian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.2️⃣.2️⃣🅱️ Special mode curve: PLOT2_2 🔖🔖 | Romanian→English 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot2_2B', default="PLOT2_Baseline_Ro_En", type=str)
exp_parser.add_argument('--mode_name_Plot2_2B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot2_2B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot2_2B', default="IWSLT17_Ro_En", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Ro' = English→Romanian, 'IWSLT17_Ro_En' = Romanian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ====================================================================================================================
# ===================== 4️⃣.3️⃣ 📌📌 ENGLISH ⬅️➡️ ITALIAN  ========================================================== 
# ====================================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.3️⃣.1️⃣🅰️ Special mode curve: PLOT3_1 🔖🔖 | English→Italian 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot3_1A', default="PLOT3_TDM_Former_En_It", type=str)
exp_parser.add_argument('--mode_name_Plot3_1A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot3_1A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot3_1A', default="IWSLT17_En_It", type=str,
    help="Choose dataset direction: 'IWSLT17_En_It' = English→Italian, 'IWSLT17_It_En' = Italian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.3️⃣.1️⃣🅱️ Special mode curve: PLOT3_1 🔖🔖 | English→Italian 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot3_1B', default="PLOT3_Baseline_En_It", type=str)
exp_parser.add_argument('--mode_name_Plot3_1B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot3_1B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot3_1B', default="IWSLT17_En_It", type=str,
    help="Choose dataset direction: 'IWSLT17_En_It' = English→Italian, 'IWSLT17_It_En' = Italian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.3️⃣.2️⃣🅰️ Special mode curve: PLOT3_2 🔖🔖 | Italian→English 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot3_2A', default="PLOT3_TDM_Former_It_En", type=str)
exp_parser.add_argument('--mode_name_Plot3_2A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot3_2A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot3_2A', default="IWSLT17_It_En", type=str,
    help="Choose dataset direction: 'IWSLT17_En_It' = English→Italian, 'IWSLT17_It_En' = Italian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.3️⃣.2️⃣🅱️ Special mode curve: PLOT3_2 🔖🔖 | Italian→English 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot3_2B', default="PLOT3_Baseline_It_En", type=str)
exp_parser.add_argument('--mode_name_Plot3_2B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot3_2B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot3_2B', default="IWSLT17_It_En", type=str,
    help="Choose dataset direction: 'IWSLT17_En_It' = English→Italian, 'IWSLT17_It_En' = Italian→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ====================================================================================================================
# ===================== 4️⃣.4️⃣ 📌📌 ENGLISH ⬅️➡️ CHINESE  ========================================================== 
# ====================================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.4️⃣.1️⃣🅰️ Special mode curve: PLOT4_1 🔖🔖 | English→Chinese 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot4_1A', default="PLOT4_TDM_Former_En_Zh", type=str)
exp_parser.add_argument('--mode_name_Plot4_1A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot4_1A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot4_1A', default="IWSLT17_En_Zh", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Zh' = English→Chinese, 'IWSLT17_Zh_En' = Chinese→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.4️⃣.1️⃣🅱️ Special mode curve: PLOT4_1 🔖🔖 | English→Chinese 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot4_1B', default="PLOT4_Baseline_En_Zh", type=str)
exp_parser.add_argument('--mode_name_Plot4_1B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot4_1B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot4_1B', default="IWSLT17_En_Zh", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Zh' = English→Chinese, 'IWSLT17_Zh_En' = Chinese→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.4️⃣.2️⃣🅰️ Special mode curve: PLOT4_2 🔖🔖 | Chinese→English 🅰️🔼 TDM_Former
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot4_2A', default="PLOT4_TDM_Former_Zh_En", type=str)
exp_parser.add_argument('--mode_name_Plot4_2A', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot4_2A', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot4_2A', default="IWSLT17_Zh_En", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Zh' = English→Chinese, 'IWSLT17_Zh_En' = Chinese→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.4️⃣.2️⃣🅱️ Special mode curve: PLOT4_2 🔖🔖 | Chinese→English 🅱️🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot4_2B', default="PLOT4_Baseline_Zh_En", type=str)
exp_parser.add_argument('--mode_name_Plot4_2B', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot4_2B', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot4_2B', default="IWSLT17_Zh_En", type=str,
    help="Choose dataset direction: 'IWSLT17_En_Zh' = English→Chinese, 'IWSLT17_Zh_En' = Chinese→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === Global font settings 📣 📣 =============================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === Global font settings 📣 📣 =============================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Global font settings === 
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # Default: 9

exp_args = exp_parser.parse_args([])   # ← for naming/logging
# ─────────────────────────────────────────────────────────────────────────────────────────────────








########################################################################################################################
####-------| NOTE 4. GLOBAL FONT| XXX --------------------------------------------------------------####################
########################################################################################################################
# ===============================================================
# 🔗============= GLOBAL FONT SETTINGs 🔑=====================🔗
# ===============================================================

plt.rcParams.update({

    # === FONT SETTINGS ===
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\boldmath",  # makes all LaTeX text bold

    # === Colors ===
    "text.color": "#000000",               # solid black
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.titlecolor": "#000000",


    "font.size": exp_args.base_font_size,
    "font.weight": "normal",
    "axes.titlesize": exp_args.base_font_size + 1,
    "axes.titleweight": "normal",
    "axes.labelsize": exp_args.base_font_size + 2,
    "axes.labelweight": "medium",
    "legend.fontsize": exp_args.base_font_size - 1,
    "legend.title_fontsize": exp_args.base_font_size,
    "xtick.labelsize": exp_args.base_font_size,
    "ytick.labelsize": exp_args.base_font_size,

    # === COLOR CONSISTENCY ===
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "axes.edgecolor": "black",


    # === AXES & SPINES ===
    "axes.linewidth": exp_args.spine_width,   # ✅ ensures ALL future figures use this spine width        
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.axisbelow": False,   # ensures lines/markers are above grid

    # === PDF / SVG EXPORT QUALITY ===
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,         # editable text in PS
    "svg.fonttype": 'none',    # editable text in SVG
})

print(f"✅ Publication style applied: Bold fonts, black ticks, clean spines (base font size={exp_args.base_font_size} | width={exp_args.spine_width}).")
# ─────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 5. SANITY CHECK| XXX -------------------------------------------------------------####################
########################################################################################################################
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##################################################


# ======================================================================================================
# ✅ =======================🔖 TEST/TRAIN ACCURACY-BLEU-LOSS 🔖========================================
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
DATA_TEST_PATH = r"./"

print("\n📂 Files actually present in ./\n")
for f in sorted(os.listdir("./")):
    print(" ", f)
# ─────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 6. READ LOG FUNCTIONS (Updated for MT logs)| XXX -----🔑🔗-----------------------####################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ === Read test loss, BLEU, best BLEU, and best epoch ===
def read_test_metrics(file_path):
    epochs, test_losses, bleu_scores = [], [], []
    best_bleu = None
    best_epoch = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, test_losses, bleu_scores, best_bleu, best_epoch

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:

            # Example:
            # Checkpoint epoch_48_TDM_Former_...pt | Test Loss: 3.149 | sacreBLEU: 25.76 | Beam=5
            match = re.search(
                r"Checkpoint\s+epoch_(\d+)_.*?\|\s*Test Loss:\s*([\d.]+)\s*\|\s*sacreBLEU:\s*([\d.]+)",
                line
            )

            if match:
                epoch = int(match.group(1))
                test_loss = float(match.group(2))
                bleu = float(match.group(3))

                epochs.append(epoch)
                test_losses.append(test_loss)
                bleu_scores.append(bleu)

                # Track best BLEU and epoch directly from checkpoint lines
                if best_bleu is None or bleu > best_bleu:
                    best_bleu = bleu
                    best_epoch = epoch

    return epochs, test_losses, bleu_scores, best_bleu, best_epoch
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ === Read train loss, learning rate, best/lower train loss, and best epoch ===
def read_train_metrics(file_path):
    epochs, train_losses, learning_rates = [], [], []
    best_train_loss = None
    best_epoch = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, train_losses, learning_rates, best_train_loss, best_epoch

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:

            # Example:
            # Epoch 48 | Train Loss: 2.993 | LR: 0.00002548
            match = re.search(
                r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\d.]+)",
                line
            )

            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                lr = float(match.group(3))

                epochs.append(epoch)
                train_losses.append(train_loss)
                learning_rates.append(lr)

                # Lower train loss is better
                if best_train_loss is None or train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_epoch = epoch

    return epochs, train_losses, learning_rates, best_train_loss, best_epoch
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣ === Compute absolute and relative BLEU gains ===
def compute_bleu_gains(tdm_bleu, base_bleu):
    abs_gain = tdm_bleu - base_bleu
    rel_gain = (abs_gain / base_bleu) * 100.0
    return abs_gain, rel_gain
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 7. DEFINE FILE PATH | XXX --------------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== Define Train/Test log file paths ==================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def build_log_path(curve_id, folder_name):
    mode_name    = getattr(exp_args, f"mode_name_Plot{curve_id}")
    net_name     = getattr(exp_args, f"net_Plot{curve_id}")
    dataset_name = getattr(exp_args, f"dataset_name_Plot{curve_id}")

    if exp_args.evaluation_mode == "test":
        file_name = f"TestCheckpoints_{net_name}_{dataset_name}_{exp_args.optimizer1}_{mode_name}.txt"

    elif exp_args.evaluation_mode == "train":
        file_name = f"Train_{net_name}_{dataset_name}_{exp_args.optimizer1}_{mode_name}.txt"

    else:
        raise ValueError(f"Unknown Evaluation mode: {exp_args.evaluation_mode}")

    return os.path.join(
        DATA_TEST_PATH,
        folder_name,
        "Results",
        dataset_name,
        net_name,
        file_name
    )

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define all log file paths ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────

Path_Plot_1A = build_log_path("1_1A", "1.1-Transformer_Baselines_EN_DE")
Path_Plot_1B = build_log_path("1_1B", "1.1-Transformer_Baselines_EN_DE")

Path_Plot_2A = build_log_path("1_2A", "1.2-Transformer_Baselines_DE_EN")
Path_Plot_2B = build_log_path("1_2B", "1.2-Transformer_Baselines_DE_EN")

Path_Plot_3A = build_log_path("2_1A", "2.1-Transformer_Baselines_EN_RO")
Path_Plot_3B = build_log_path("2_1B", "2.1-Transformer_Baselines_EN_RO")

Path_Plot_4A = build_log_path("2_2A", "2.2-Transformer_Baselines_RO_EN")
Path_Plot_4B = build_log_path("2_2B", "2.2-Transformer_Baselines_RO_EN")

Path_Plot_5A = build_log_path("3_1A", "3.1-Transformer_Baselines_EN_IT")
Path_Plot_5B = build_log_path("3_1B", "3.1-Transformer_Baselines_EN_IT")

Path_Plot_6A = build_log_path("3_2A", "3.2-Transformer_Baselines_IT_EN")
Path_Plot_6B = build_log_path("3_2B", "3.2-Transformer_Baselines_IT_EN")

Path_Plot_7A = build_log_path("4_1A", "4.1-Transformer_Baselines_EN_ZH")
Path_Plot_7B = build_log_path("4_1B", "4.1-Transformer_Baselines_EN_ZH")

Path_Plot_8A = build_log_path("4_2A", "4.2-Transformer_Baselines_ZH_EN")
Path_Plot_8B = build_log_path("4_2B", "4.2-Transformer_Baselines_ZH_EN")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print log file paths sanity check ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────

print(f"\n📁 {exp_args.evaluation_mode.upper()} log file paths:")
print("─" * 100)

all_paths = {
    "EN→DE | TDM":      Path_Plot_1A,
    "EN→DE | Baseline": Path_Plot_1B,

    "DE→EN | TDM":      Path_Plot_2A,
    "DE→EN | Baseline": Path_Plot_2B,

    "EN→RO | TDM":      Path_Plot_3A,
    "EN→RO | Baseline": Path_Plot_3B,

    "RO→EN | TDM":      Path_Plot_4A,
    "RO→EN | Baseline": Path_Plot_4B,

    "EN→IT | TDM":      Path_Plot_5A,
    "EN→IT | Baseline": Path_Plot_5B,

    "IT→EN | TDM":      Path_Plot_6A,
    "IT→EN | Baseline": Path_Plot_6B,

    "EN→ZH | TDM":      Path_Plot_7A,
    "EN→ZH | Baseline": Path_Plot_7B,

    "ZH→EN | TDM":      Path_Plot_8A,
    "ZH→EN | Baseline": Path_Plot_8B,
}

for name, path in all_paths.items():
    print(f"🧪 {name}:")
    print(f"   {path}\n")

print("─" * 100)
# ─────────────────────────────────────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 8. READ LOGS| XXX -----🔑🔗------------------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 8. READ LOGS| XXX -----🔑🔗------------------------------------------------------####################
########################################################################################################################

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ 🧩🧠 === Test ♻️ ♻️ ======================================================================= 
# ────────────────────────────────────────────────────────────────────────────────────────────────
if exp_args.evaluation_mode == "test": 

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ === Read TEST metrics: Test Loss + BLEU + Best BLEU + Best Epoch
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    EN_DE_TDM_epochs_test, EN_DE_TDM_test_loss, EN_DE_TDM_bleu, EN_DE_TDM_best_bleu, EN_DE_TDM_best_epoch = read_test_metrics(Path_Plot_1A)
    EN_DE_BASE_epochs_test, EN_DE_BASE_test_loss, EN_DE_BASE_bleu, EN_DE_BASE_best_bleu, EN_DE_BASE_best_epoch = read_test_metrics(Path_Plot_1B)

    DE_EN_TDM_epochs_test, DE_EN_TDM_test_loss, DE_EN_TDM_bleu, DE_EN_TDM_best_bleu, DE_EN_TDM_best_epoch = read_test_metrics(Path_Plot_2A)
    DE_EN_BASE_epochs_test, DE_EN_BASE_test_loss, DE_EN_BASE_bleu, DE_EN_BASE_best_bleu, DE_EN_BASE_best_epoch = read_test_metrics(Path_Plot_2B)

    EN_RO_TDM_epochs_test, EN_RO_TDM_test_loss, EN_RO_TDM_bleu, EN_RO_TDM_best_bleu, EN_RO_TDM_best_epoch = read_test_metrics(Path_Plot_3A)
    EN_RO_BASE_epochs_test, EN_RO_BASE_test_loss, EN_RO_BASE_bleu, EN_RO_BASE_best_bleu, EN_RO_BASE_best_epoch = read_test_metrics(Path_Plot_3B)

    RO_EN_TDM_epochs_test, RO_EN_TDM_test_loss, RO_EN_TDM_bleu, RO_EN_TDM_best_bleu, RO_EN_TDM_best_epoch = read_test_metrics(Path_Plot_4A)
    RO_EN_BASE_epochs_test, RO_EN_BASE_test_loss, RO_EN_BASE_bleu, RO_EN_BASE_best_bleu, RO_EN_BASE_best_epoch = read_test_metrics(Path_Plot_4B)

    EN_IT_TDM_epochs_test, EN_IT_TDM_test_loss, EN_IT_TDM_bleu, EN_IT_TDM_best_bleu, EN_IT_TDM_best_epoch = read_test_metrics(Path_Plot_5A)
    EN_IT_BASE_epochs_test, EN_IT_BASE_test_loss, EN_IT_BASE_bleu, EN_IT_BASE_best_bleu, EN_IT_BASE_best_epoch = read_test_metrics(Path_Plot_5B)

    IT_EN_TDM_epochs_test, IT_EN_TDM_test_loss, IT_EN_TDM_bleu, IT_EN_TDM_best_bleu, IT_EN_TDM_best_epoch = read_test_metrics(Path_Plot_6A)
    IT_EN_BASE_epochs_test, IT_EN_BASE_test_loss, IT_EN_BASE_bleu, IT_EN_BASE_best_bleu, IT_EN_BASE_best_epoch = read_test_metrics(Path_Plot_6B)

    EN_ZH_TDM_epochs_test, EN_ZH_TDM_test_loss, EN_ZH_TDM_bleu, EN_ZH_TDM_best_bleu, EN_ZH_TDM_best_epoch = read_test_metrics(Path_Plot_7A)
    EN_ZH_BASE_epochs_test, EN_ZH_BASE_test_loss, EN_ZH_BASE_bleu, EN_ZH_BASE_best_bleu, EN_ZH_BASE_best_epoch = read_test_metrics(Path_Plot_7B)

    ZH_EN_TDM_epochs_test, ZH_EN_TDM_test_loss, ZH_EN_TDM_bleu, ZH_EN_TDM_best_bleu, ZH_EN_TDM_best_epoch = read_test_metrics(Path_Plot_8A)
    ZH_EN_BASE_epochs_test, ZH_EN_BASE_test_loss, ZH_EN_BASE_bleu, ZH_EN_BASE_best_bleu, ZH_EN_BASE_best_epoch = read_test_metrics(Path_Plot_8B)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🏆 Print TEST summary with BLEU gain
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    print("\n🏆 TEST Best BLEU Summary:")
    print("─" * 110)
    print(f"{'Task':<8} | {'Baseline':>8} | {'TDM':>8} | {'Gain':>8} | {'Rel. Gain':>10} | {'TDM Epoch':>9} | {'Base Epoch':>10}")
    print("─" * 110)

    summary_rows = [
        ("EN→DE", EN_DE_TDM_best_bleu, EN_DE_BASE_best_bleu, EN_DE_TDM_best_epoch, EN_DE_BASE_best_epoch),
        ("DE→EN", DE_EN_TDM_best_bleu, DE_EN_BASE_best_bleu, DE_EN_TDM_best_epoch, DE_EN_BASE_best_epoch),
        ("EN→RO", EN_RO_TDM_best_bleu, EN_RO_BASE_best_bleu, EN_RO_TDM_best_epoch, EN_RO_BASE_best_epoch),
        ("RO→EN", RO_EN_TDM_best_bleu, RO_EN_BASE_best_bleu, RO_EN_TDM_best_epoch, RO_EN_BASE_best_epoch),
        ("EN→IT", EN_IT_TDM_best_bleu, EN_IT_BASE_best_bleu, EN_IT_TDM_best_epoch, EN_IT_BASE_best_epoch),
        ("IT→EN", IT_EN_TDM_best_bleu, IT_EN_BASE_best_bleu, IT_EN_TDM_best_epoch, IT_EN_BASE_best_epoch),
        ("EN→ZH", EN_ZH_TDM_best_bleu, EN_ZH_BASE_best_bleu, EN_ZH_TDM_best_epoch, EN_ZH_BASE_best_epoch),
        ("ZH→EN", ZH_EN_TDM_best_bleu, ZH_EN_BASE_best_bleu, ZH_EN_TDM_best_epoch, ZH_EN_BASE_best_epoch),
    ]

    for task, tdm_bleu, base_bleu, tdm_epoch, base_epoch in summary_rows:
        # abs_gain, rel_gain = bleu_gain(tdm_bleu, base_bleu)
        abs_gain, rel_gain = compute_bleu_gains(tdm_bleu, base_bleu)

        print(
            f"{task:<8} | "
            f"{base_bleu:>8.2f} | "
            f"{tdm_bleu:>8.2f} | "
            f"{abs_gain:>+8.2f} | "
            f"{rel_gain:>+9.2f}% | "
            f"{tdm_epoch:>9} | "
            f"{base_epoch:>10}"
        )

    print("─" * 110)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 💾 Save TEST summary to TXT file
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    summary_txt_path = os.path.join(DATA_TEST_PATH, "TDM_Former_vs_Fairseq_Baseline_TEST_BLEU_Gains.txt")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("🏆 TEST Best BLEU Summary\n")
        f.write("─" * 110 + "\n")
        f.write(f"{'Task':<8} | {'Baseline':>8} | {'TDM':>8} | {'Gain':>8} | {'Rel. Gain':>10} | {'TDM Epoch':>9} | {'Base Epoch':>10}\n")
        f.write("─" * 110 + "\n")

        for task, tdm_bleu, base_bleu, tdm_epoch, base_epoch in summary_rows:
            abs_gain, rel_gain = compute_bleu_gains(tdm_bleu, base_bleu)

            f.write(
                f"{task:<8} | "
                f"{base_bleu:>8.2f} | "
                f"{tdm_bleu:>8.2f} | "
                f"{abs_gain:>+8.2f} | "
                f"{rel_gain:>+9.2f}% | "
                f"{tdm_epoch:>9} | "
                f"{base_epoch:>10}\n"
            )

        f.write("─" * 110 + "\n")

    print(f"✅ Saved TEST BLEU summary to: {summary_txt_path}")



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ 🧩🧠 === Train ♻️ ♻️ ======================================================================= 
# ────────────────────────────────────────────────────────────────────────────────────────────────
elif exp_args.evaluation_mode == "train": 

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ === Read TRAIN metrics: Train Loss + LR + Best/Lowest Train Loss + Best Epoch
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    EN_DE_TDM_epochs_train, EN_DE_TDM_train_loss, EN_DE_TDM_lr, EN_DE_TDM_best_loss, EN_DE_TDM_best_epoch = read_train_metrics(Path_Plot_1A)
    EN_DE_BASE_epochs_train, EN_DE_BASE_train_loss, EN_DE_BASE_lr, EN_DE_BASE_best_loss, EN_DE_BASE_best_epoch = read_train_metrics(Path_Plot_1B)

    DE_EN_TDM_epochs_train, DE_EN_TDM_train_loss, DE_EN_TDM_lr, DE_EN_TDM_best_loss, DE_EN_TDM_best_epoch = read_train_metrics(Path_Plot_2A)
    DE_EN_BASE_epochs_train, DE_EN_BASE_train_loss, DE_EN_BASE_lr, DE_EN_BASE_best_loss, DE_EN_BASE_best_epoch = read_train_metrics(Path_Plot_2B)

    EN_RO_TDM_epochs_train, EN_RO_TDM_train_loss, EN_RO_TDM_lr, EN_RO_TDM_best_loss, EN_RO_TDM_best_epoch = read_train_metrics(Path_Plot_3A)
    EN_RO_BASE_epochs_train, EN_RO_BASE_train_loss, EN_RO_BASE_lr, EN_RO_BASE_best_loss, EN_RO_BASE_best_epoch = read_train_metrics(Path_Plot_3B)

    RO_EN_TDM_epochs_train, RO_EN_TDM_train_loss, RO_EN_TDM_lr, RO_EN_TDM_best_loss, RO_EN_TDM_best_epoch = read_train_metrics(Path_Plot_4A)
    RO_EN_BASE_epochs_train, RO_EN_BASE_train_loss, RO_EN_BASE_lr, RO_EN_BASE_best_loss, RO_EN_BASE_best_epoch = read_train_metrics(Path_Plot_4B)

    EN_IT_TDM_epochs_train, EN_IT_TDM_train_loss, EN_IT_TDM_lr, EN_IT_TDM_best_loss, EN_IT_TDM_best_epoch = read_train_metrics(Path_Plot_5A)
    EN_IT_BASE_epochs_train, EN_IT_BASE_train_loss, EN_IT_BASE_lr, EN_IT_BASE_best_loss, EN_IT_BASE_best_epoch = read_train_metrics(Path_Plot_5B)

    IT_EN_TDM_epochs_train, IT_EN_TDM_train_loss, IT_EN_TDM_lr, IT_EN_TDM_best_loss, IT_EN_TDM_best_epoch = read_train_metrics(Path_Plot_6A)
    IT_EN_BASE_epochs_train, IT_EN_BASE_train_loss, IT_EN_BASE_lr, IT_EN_BASE_best_loss, IT_EN_BASE_best_epoch = read_train_metrics(Path_Plot_6B)

    EN_ZH_TDM_epochs_train, EN_ZH_TDM_train_loss, EN_ZH_TDM_lr, EN_ZH_TDM_best_loss, EN_ZH_TDM_best_epoch = read_train_metrics(Path_Plot_7A)
    EN_ZH_BASE_epochs_train, EN_ZH_BASE_train_loss, EN_ZH_BASE_lr, EN_ZH_BASE_best_loss, EN_ZH_BASE_best_epoch = read_train_metrics(Path_Plot_7B)

    ZH_EN_TDM_epochs_train, ZH_EN_TDM_train_loss, ZH_EN_TDM_lr, ZH_EN_TDM_best_loss, ZH_EN_TDM_best_epoch = read_train_metrics(Path_Plot_8A)
    ZH_EN_BASE_epochs_train, ZH_EN_BASE_train_loss, ZH_EN_BASE_lr, ZH_EN_BASE_best_loss, ZH_EN_BASE_best_epoch = read_train_metrics(Path_Plot_8B)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🏆 Print TRAIN summary
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    print("\n🏆 TRAIN Best/Lowest Loss Summary:")
    print("─" * 90)

    print(f"EN→DE | TDM      : Loss={EN_DE_TDM_best_loss:.3f} | Epoch={EN_DE_TDM_best_epoch}")
    print(f"EN→DE | Baseline : Loss={EN_DE_BASE_best_loss:.3f} | Epoch={EN_DE_BASE_best_epoch}")

    print(f"DE→EN | TDM      : Loss={DE_EN_TDM_best_loss:.3f} | Epoch={DE_EN_TDM_best_epoch}")
    print(f"DE→EN | Baseline : Loss={DE_EN_BASE_best_loss:.3f} | Epoch={DE_EN_BASE_best_epoch}")

    print(f"EN→RO | TDM      : Loss={EN_RO_TDM_best_loss:.3f} | Epoch={EN_RO_TDM_best_epoch}")
    print(f"EN→RO | Baseline : Loss={EN_RO_BASE_best_loss:.3f} | Epoch={EN_RO_BASE_best_epoch}")

    print(f"RO→EN | TDM      : Loss={RO_EN_TDM_best_loss:.3f} | Epoch={RO_EN_TDM_best_epoch}")
    print(f"RO→EN | Baseline : Loss={RO_EN_BASE_best_loss:.3f} | Epoch={RO_EN_BASE_best_epoch}")

    print(f"EN→IT | TDM      : Loss={EN_IT_TDM_best_loss:.3f} | Epoch={EN_IT_TDM_best_epoch}")
    print(f"EN→IT | Baseline : Loss={EN_IT_BASE_best_loss:.3f} | Epoch={EN_IT_BASE_best_epoch}")

    print(f"IT→EN | TDM      : Loss={IT_EN_TDM_best_loss:.3f} | Epoch={IT_EN_TDM_best_epoch}")
    print(f"IT→EN | Baseline : Loss={IT_EN_BASE_best_loss:.3f} | Epoch={IT_EN_BASE_best_epoch}")

    print(f"EN→ZH | TDM      : Loss={EN_ZH_TDM_best_loss:.3f} | Epoch={EN_ZH_TDM_best_epoch}")
    print(f"EN→ZH | Baseline : Loss={EN_ZH_BASE_best_loss:.3f} | Epoch={EN_ZH_BASE_best_epoch}")

    print(f"ZH→EN | TDM      : Loss={ZH_EN_TDM_best_loss:.3f} | Epoch={ZH_EN_TDM_best_epoch}")
    print(f"ZH→EN | Baseline : Loss={ZH_EN_BASE_best_loss:.3f} | Epoch={ZH_EN_BASE_best_epoch}")

    print("─" * 90)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(f"Unknown Evaluation mode: {exp_args.evaluation_mode}")
# =====================================================================================================




# %% 

########################################################################################################################
####-------| NOTE 9. PLOT FUNCTION | XXX -----🔑📐🔗-----------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9. PLOT FUNCTION | XXX -----🔑📐🔗-----------------------------------------------####################
########################################################################################################################
# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================
# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

from matplotlib.lines import Line2D

def plot_tdm_bleu_vs_epoch_4pairs(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅰️🔼 🎨 === Colors / styles
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    COLORS = {
        "TDM_forward":      "#E49B0F",
        "BASE_forward":     "#EF476F",
        "TDM_reverse":      "#8338EC",
        "BASE_reverse":     "#06D6A0",
    }


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅱️🔼🔧 === One figure = one language pair, both directions in same axes
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def plot_language_pair_same_axes(
        pair_title,
        file_name,

        forward_label,
        reverse_label,

        forward_tdm_epochs,
        forward_tdm_bleu,
        forward_base_epochs,
        forward_base_bleu,

        reverse_tdm_epochs,
        reverse_tdm_bleu,
        reverse_base_epochs,
        reverse_base_bleu,

        forward_tdm_best_bleu,
        forward_tdm_best_epoch,
        forward_base_best_bleu,
        forward_base_best_epoch,

        reverse_tdm_best_bleu,
        reverse_tdm_best_epoch,
        reverse_base_best_bleu,
        reverse_base_best_epoch
    ):

        fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

        # ────────────────────────────────────────────────────────────────
        # 1️⃣.1️⃣ ========= Forward direction =============================
        # ────────────────────────────────────────────────────────────────
        ax.plot(
            forward_tdm_epochs,
            forward_tdm_bleu,
            label=rf"\textbf{{{forward_label} TDM-Former}}",
            color=COLORS["TDM_forward"],
            linewidth=2.2,
            linestyle="-",
            alpha=1.0,
            zorder=6
        )

        ax.plot(
            forward_base_epochs,
            forward_base_bleu,
            label=rf"\textbf{{{forward_label} Baseline}}",
            color=COLORS["BASE_forward"],
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=5
        )

        # ────────────────────────────────────────────────────────────────
        # 1️⃣.1️⃣ ========= Reverse direction =============================
        # ────────────────────────────────────────────────────────────────
        ax.plot(
            reverse_tdm_epochs,
            reverse_tdm_bleu,
            label=rf"\textbf{{{reverse_label} TDM-Former}}",
            color=COLORS["TDM_reverse"],
            linewidth=2.2,
            linestyle="-",
            alpha=1.0,
            zorder=6
        )

        ax.plot(
            reverse_base_epochs,
            reverse_base_bleu,
            label=rf"\textbf{{{reverse_label} Baseline}}",
            color=COLORS["BASE_reverse"],
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=5
        )

        # ────────────────────────────────────────────────────────────────
        # 2️⃣📉 ========= Mark best BLEU points ==========================
        # ────────────────────────────────────────────────────────────────
        # ax.scatter([forward_tdm_best_epoch], [forward_tdm_best_bleu], color=COLORS["TDM_forward"], s=35, zorder=10)
        # ax.scatter([forward_base_best_epoch], [forward_base_best_bleu], color=COLORS["BASE_forward"], s=35, zorder=10)
        # ax.scatter([reverse_tdm_best_epoch], [reverse_tdm_best_bleu], color=COLORS["TDM_reverse"], s=35, zorder=10)
        # ax.scatter([reverse_base_best_epoch], [reverse_base_best_bleu], color=COLORS["BASE_reverse"], s=35, zorder=10)


        ax.scatter([forward_tdm_best_epoch], [forward_tdm_best_bleu], color=COLORS["TDM_forward"], 
                   s=30, facecolors='white', edgecolors=COLORS["TDM_forward"], linewidths=1.6, zorder=10)
        ax.scatter([forward_base_best_epoch], [forward_base_best_bleu], color=COLORS["BASE_forward"], 
                   s=30, facecolors='white', edgecolors=COLORS["BASE_forward"], linewidths=1.6, zorder=10)
        ax.scatter([reverse_tdm_best_epoch], [reverse_tdm_best_bleu], color=COLORS["TDM_reverse"], 
                   s=30, facecolors='white', edgecolors=COLORS["TDM_reverse"], linewidths=1.6, zorder=10)
        ax.scatter([reverse_base_best_epoch], [reverse_base_best_bleu], color=COLORS["BASE_reverse"], 
                   s=30, facecolors='white', edgecolors=COLORS["BASE_reverse"], linewidths=1.6, zorder=10)    


        # ────────────────────────────────────────────────────────────────
        # 3️⃣.1️⃣⚙️ ========= Axis labels / title =========================
        # ────────────────────────────────────────────────────────────────
        # ax.set_title(r"\textbf{" + pair_title + r"}")
        ax.set_xlabel(r"\textbf{Epoch}")
        ax.set_ylabel(r"\textbf{BLEU}")
        ax.grid(True, linestyle="--", alpha=0.35)

        # ────────────────────────────────────────────────────────────────
        # 3️⃣.2️⃣⚙️ ========= Axis control ===============================
        # ────────────────────────────────────────────────────────────────
        # ax.set_xlim(-2, 51)
        ax.set_xlim(-1, 51)
        # ax.set_xticks(range(0, 50, 10))
        ax.set_xticks([0, 10, 20, 30, 40, 50])

        ax.set_ylim(19, 36)
        ax.set_yticks([20, 25, 30, 35])
        # ────────────────────────────────────────────────────────────────


        # ────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────
        # 4️⃣.1️⃣🔍♻️ ======== GROUPED LEGEND: Forward | Reverse =========
        # ────────────────────────────────────────────────────────────────
        legend_handles = [

            # 🔧 ---- Column 1: Forward direction ----
            Line2D([0], [0],
                color=COLORS["TDM_forward"],
                linestyle="-",
                linewidth=2.2,
                # label="TDM-Former"),
                label="TDM"),

            Line2D([0], [0],
                color=COLORS["BASE_forward"],
                linestyle="--",
                linewidth=2.0,
                # label="Baseline"),
                label="Base"),
            #----------------------------------------------------------------
            # 🔧 ---- Column 2: Reverse direction ----
            Line2D([0], [0],
                color=COLORS["TDM_reverse"],
                linestyle="-",
                linewidth=2.2,
                # label="TDM-Former"),
                label="TDM"),

            Line2D([0], [0],
                color=COLORS["BASE_reverse"],
                linestyle="--",
                linewidth=2.0,
                # label="Baseline"),
                label="Base"),
        ]

        # ────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────
        # 4️⃣.2️⃣🔍♻️ ===== CREATE GROUPED LEGEND: Forward | Revers ======
        # ────────────────────────────────────────────────────────────────
        leg = ax.legend(
            handles=legend_handles,
            ncol=2,
            frameon=False,
            loc="upper left",
            # bbox_to_anchor=(0.02, 0.98),
            bbox_to_anchor=(0.00, 0.94),
            columnspacing=1.0,
            handlelength=1.2,
            handletextpad=0.35,
            labelspacing=0.35,
            borderaxespad=0.2,
            fontsize=exp_args.legend_font
        )        

        leg._legend_box.align = "center"
        leg.set_zorder(100)
        #----------------------------------------------------------------
        # 4️⃣.2️⃣.1️⃣🔧 ---- Bold legend entries ----
        #----------------------------------------------------------------
        for t in leg.get_texts():
            t.set_text(r"\textbf{" + t.get_text() + "}")
        #----------------------------------------------------------------
        # 4️⃣.2️⃣.2️⃣🔧🏷️ --- Add legend column titles: Forward | Reverse
        #----------------------------------------------------------------
        ax.text(
            0.10, 0.95,
            r"\textbf{" + forward_label + "}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            # fontsize=exp_args.legend_title_font
            fontsize=exp_args.legend_title_font + 0.5
        )

        ax.text(
            0.305, 0.95,
            r"\textbf{" + reverse_label + "}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            # fontsize=exp_args.legend_title_font
            fontsize=exp_args.legend_title_font + 0.5
        )
        # ────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────


        # ────────────────────────────────────────────────────────────────
        # 5️⃣📦 ======== Save figures ====================================
        # ────────────────────────────────────────────────────────────────
        # Save
        fig.savefig(
            os.path.join(save_dir, f"{file_name}_BLEU_vs_Epoch.pdf"),
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            dpi=600
        )

        fig.savefig(
            os.path.join(save_dir, f"{file_name}_BLEU_vs_Epoch.svg"),
            format="svg",
            bbox_inches="tight",
            facecolor="white"
        )

        plt.show()


    # =================================================================================================
    # 1️⃣ 📌📌 ENGLISH ⬅️➡️ GERMAN
    # =================================================================================================

    plot_language_pair_same_axes(
        pair_title=r"English $\leftrightarrow$ German",
        file_name="Plot1_EN_DE_DE_EN_TDM_vs_Fairseq_Baseline",

        forward_label=r"EN$\rightarrow$DE",
        reverse_label=r"DE$\rightarrow$EN",

        forward_tdm_epochs=EN_DE_TDM_epochs_test,
        forward_tdm_bleu=EN_DE_TDM_bleu,
        forward_base_epochs=EN_DE_BASE_epochs_test,
        forward_base_bleu=EN_DE_BASE_bleu,

        reverse_tdm_epochs=DE_EN_TDM_epochs_test,
        reverse_tdm_bleu=DE_EN_TDM_bleu,
        reverse_base_epochs=DE_EN_BASE_epochs_test,
        reverse_base_bleu=DE_EN_BASE_bleu,

        forward_tdm_best_bleu=EN_DE_TDM_best_bleu,
        forward_tdm_best_epoch=EN_DE_TDM_best_epoch,
        forward_base_best_bleu=EN_DE_BASE_best_bleu,
        forward_base_best_epoch=EN_DE_BASE_best_epoch,

        reverse_tdm_best_bleu=DE_EN_TDM_best_bleu,
        reverse_tdm_best_epoch=DE_EN_TDM_best_epoch,
        reverse_base_best_bleu=DE_EN_BASE_best_bleu,
        reverse_base_best_epoch=DE_EN_BASE_best_epoch
    )


    # =================================================================================================
    # 2️⃣ 📌📌 ENGLISH ⬅️➡️ ROMANIAN
    # =================================================================================================

    plot_language_pair_same_axes(
        pair_title=r"English $\leftrightarrow$ Romanian",
        file_name="Plot2_EN_RO_RO_EN_TDM_vs_Fairseq_Baseline",

        forward_label=r"EN$\rightarrow$RO",
        reverse_label=r"RO$\rightarrow$EN",

        forward_tdm_epochs=EN_RO_TDM_epochs_test,
        forward_tdm_bleu=EN_RO_TDM_bleu,
        forward_base_epochs=EN_RO_BASE_epochs_test,
        forward_base_bleu=EN_RO_BASE_bleu,

        reverse_tdm_epochs=RO_EN_TDM_epochs_test,
        reverse_tdm_bleu=RO_EN_TDM_bleu,
        reverse_base_epochs=RO_EN_BASE_epochs_test,
        reverse_base_bleu=RO_EN_BASE_bleu,

        forward_tdm_best_bleu=EN_RO_TDM_best_bleu,
        forward_tdm_best_epoch=EN_RO_TDM_best_epoch,
        forward_base_best_bleu=EN_RO_BASE_best_bleu,
        forward_base_best_epoch=EN_RO_BASE_best_epoch,

        reverse_tdm_best_bleu=RO_EN_TDM_best_bleu,
        reverse_tdm_best_epoch=RO_EN_TDM_best_epoch,
        reverse_base_best_bleu=RO_EN_BASE_best_bleu,
        reverse_base_best_epoch=RO_EN_BASE_best_epoch
    )


    # =================================================================================================
    # 3️⃣ 📌📌 ENGLISH ⬅️➡️ ITALIAN
    # =================================================================================================

    plot_language_pair_same_axes(
        pair_title=r"English $\leftrightarrow$ Italian",
        file_name="Plot3_EN_IT_IT_EN_TDM_vs_Fairseq_Baseline",

        forward_label=r"EN$\rightarrow$IT",
        reverse_label=r"IT$\rightarrow$EN",

        forward_tdm_epochs=EN_IT_TDM_epochs_test,
        forward_tdm_bleu=EN_IT_TDM_bleu,
        forward_base_epochs=EN_IT_BASE_epochs_test,
        forward_base_bleu=EN_IT_BASE_bleu,

        reverse_tdm_epochs=IT_EN_TDM_epochs_test,
        reverse_tdm_bleu=IT_EN_TDM_bleu,
        reverse_base_epochs=IT_EN_BASE_epochs_test,
        reverse_base_bleu=IT_EN_BASE_bleu,

        forward_tdm_best_bleu=EN_IT_TDM_best_bleu,
        forward_tdm_best_epoch=EN_IT_TDM_best_epoch,
        forward_base_best_bleu=EN_IT_BASE_best_bleu,
        forward_base_best_epoch=EN_IT_BASE_best_epoch,

        reverse_tdm_best_bleu=IT_EN_TDM_best_bleu,
        reverse_tdm_best_epoch=IT_EN_TDM_best_epoch,
        reverse_base_best_bleu=IT_EN_BASE_best_bleu,
        reverse_base_best_epoch=IT_EN_BASE_best_epoch
    )


    # =================================================================================================
    # 4️⃣ 📌📌 ENGLISH ⬅️➡️ CHINESE
    # =================================================================================================

    plot_language_pair_same_axes(
        pair_title=r"English $\leftrightarrow$ Chinese",
        file_name="Plot4_EN_ZH_ZH_EN_TDM_vs_Fairseq_Baseline",

        forward_label=r"EN$\rightarrow$ZH",
        reverse_label=r"ZH$\rightarrow$EN",

        forward_tdm_epochs=EN_ZH_TDM_epochs_test,
        forward_tdm_bleu=EN_ZH_TDM_bleu,
        forward_base_epochs=EN_ZH_BASE_epochs_test,
        forward_base_bleu=EN_ZH_BASE_bleu,

        reverse_tdm_epochs=ZH_EN_TDM_epochs_test,
        reverse_tdm_bleu=ZH_EN_TDM_bleu,
        reverse_base_epochs=ZH_EN_BASE_epochs_test,
        reverse_base_bleu=ZH_EN_BASE_bleu,

        forward_tdm_best_bleu=EN_ZH_TDM_best_bleu,
        forward_tdm_best_epoch=EN_ZH_TDM_best_epoch,
        forward_base_best_bleu=EN_ZH_BASE_best_bleu,
        forward_base_best_epoch=EN_ZH_BASE_best_epoch,

        reverse_tdm_best_bleu=ZH_EN_TDM_best_bleu,
        reverse_tdm_best_epoch=ZH_EN_TDM_best_epoch,
        reverse_base_best_bleu=ZH_EN_BASE_best_bleu,
        reverse_base_best_epoch=ZH_EN_BASE_best_epoch
    )

# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
plot_tdm_bleu_vs_epoch_4pairs()
# ────────────────────────────────────────────────────────────────
# %% 








# %%



