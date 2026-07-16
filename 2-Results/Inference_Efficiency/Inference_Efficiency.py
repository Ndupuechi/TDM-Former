



# %% Imports and Setup


# ============= NOTE INFERENCE EFFICIENCY NOTE ==================
# 🔗==================== ANALYSIS 🔑==========================🔗
# 🔗==============⚖️ IWSLT DATASET ===========================🔗
# ===============================================================
# ============= NOTE INFERENCE EFFICIENCY NOTE ==================
# 🔗==================== ANALYSIS 🔑==========================🔗
# 🔗==============⚖️ IWSLT DATASET ===========================🔗
# ===============================================================


########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################
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
########################################################################################################################
####-------| NOTE 2. DEFINE DIRECTORY PATH | XXX ---------------------------------------------------####################
########################################################################################################################

# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Transformers\Github\2-Results\Inference_Efficiency"
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
########################################################################################################################
####-------| NOTE 3. DEFINE PARSAR DATA| XXX -----------------------------------------------------######################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 0️⃣ === Dataset information === 
exp_parser = argparse.ArgumentParser("IWSLT Ablation Config")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 === Evaluation mode ===
exp_parser.add_argument('--evaluation_mode', default="efficiency", type=str, help="Choose dataset: [test, train, efficiency]")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📦 === Dataset name ===
exp_parser.add_argument('--dataset_name', default="IWSLT14_De_En", type=str)
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
# 4️⃣⚙️ === Define curves : SPECIFIC 📣 📣 GERMAN ➡️ ENGLISH  ====================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣⚙️ === Define curves : SPECIFIC 📣 📣 GERMAN ➡️ ENGLISH  ====================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.1️⃣ Special mode curve: PLOT1 🔖🔖 | German→English 🔼 Baseline (Fairseq)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot1', default="PLOT1_Baseline_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot1', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument('--net_Plot1', default="Transformer", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot1', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.2️⃣ Special mode curve: PLOT2 🔖🔖 | German→English 🔼 TDM_Former (L6)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot2', default="PLOT2_TDM_Former_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot2', default="Seed1_1_TDM_NoRotation_Last1Layer", type=str)
exp_parser.add_argument('--net_Plot2', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot2', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.3️⃣ Special mode curve: PLOT3 🔖🔖 | German→English 🔼 TDM_Former (L5-L6)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot3', default="PLOT3_TDM_Former_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot3', default="Seed1_1_TDM_NoRotation_Last2Layer", type=str)
exp_parser.add_argument('--net_Plot3', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot3', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.4️⃣ Special mode curve: PLOT4 🔖🔖 | German→English 🔼 TDM_Former (L4-L6)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot4', default="PLOT4_TDM_Former_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot4', default="Seed1_1_TDM_NoRotation_Last3Layer", type=str)
exp_parser.add_argument('--net_Plot4', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot4', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣.5️⃣ Special mode curve: PLOT5 🔖🔖 | German→English 🔼 TDM_Former (L3-L6)
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
exp_parser.add_argument('--Plot5', default="PLOT5_TDM_Former_De_En", type=str)
exp_parser.add_argument('--mode_name_Plot5', default="Seed1_1_TDM_NoRotation_Last4Layer", type=str)
exp_parser.add_argument('--net_Plot5', default="TDM_Former", type=str)
# -------------------------------------------------------------------------------------------------
exp_parser.add_argument('--dataset_name_Plot5', default="IWSLT14_De_En", type=str,
    help="Choose dataset direction: 'IWSLT14_En_De' = English→German, 'IWSLT14_De_En' = German→English")
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣⚙️ === Global font settings 📣 📣 ===========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣⚙️ === Global font settings 📣 📣 ===========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Global font settings === 
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # Default: 9
exp_parser.add_argument('--annotation_font', default=12, type=int)

exp_args = exp_parser.parse_args([])   # ← for naming/logging
# ─────────────────────────────────────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 4. GLOBAL FONT| XXX --------------------------------------------------------------####################
########################################################################################################################
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
########################################################################################################################
####-------| NOTE 6. READ LOG FUNCTIONS (Updated for MT logs)| XXX -----🔑🔗-----------------------####################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ === READ TEST LOSS, BLEU, BEST BLEU, AND BEST EPOCH =========================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ === READ TEST LOSS, BLEU, BEST BLEU, AND BEST EPOCH =========================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

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
# 2️⃣ === READ TRAIN LOSS, LEARNING RATE, BEST/LOWER TRAIN LOSS, AND BEST EPOCH ====================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ === READ TRAIN LOSS, LEARNING RATE, BEST/LOWER TRAIN LOSS, AND BEST EPOCH ====================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

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
# 3️⃣ === COMPUTE ABSOLUTE AND RELATIVE BLEU GAINS ================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣ === COMPUTE ABSOLUTE AND RELATIVE BLEU GAINS ================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def compute_bleu_gains(tdm_bleu, base_bleu):
    abs_gain = tdm_bleu - base_bleu
    rel_gain = (abs_gain / base_bleu) * 100.0
    return abs_gain, rel_gain
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣ === READ MODEL SUMMARY =======================================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣ === READ MODEL SUMMARY =======================================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

import os
import re


def read_model_summary(file_path):
    """
    Read model_summary_full.txt and return a structured dictionary.

    Extracts:
      1) Dataset Information
      2) Complexity: Params, MACs, FLOPs
      3) Optimiser & Activation
      4) Fairseq Transformer parameters
      5) TDM parameters, only if model is TDM_Former
    """

    if not os.path.exists(file_path):
        print(f"⚠️ Missing model summary file: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # Helper functions
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def search_value(pattern, default=None, cast=str):
        match = re.search(pattern, text)
        if not match:
            return default

        value = match.group(1).strip()

        if cast == int:
            return int(value.replace(",", ""))
        elif cast == float:
            return float(value.replace(",", ""))
        elif cast == bool:
            return value.lower() == "true"
        else:
            return value

    def search_complexity(prefix_name=None):
        """
        Finds first Params/MACs/FLOPs line by default.
        If prefix_name is provided, searches near that model section.
        """
        pattern = r"📦 Params:\s*([\d.]+)\s*M\s*\|\s*⚙️ MACs:\s*([\d.]+)\s*GMACs\s*\|\s*🔥 FLOPs:\s*([\d.]+)\s*GFLOPS"

        if prefix_name is None:
            match = re.search(pattern, text)
        else:
            section_pattern = rf"{re.escape(prefix_name)}.*?{pattern}"
            match = re.search(section_pattern, text, flags=re.DOTALL)

        if not match:
            return {
                "params_M": None,
                "macs_G": None,
                "flops_G": None,
            }

        return {
            "params_M": float(match.group(1)),
            "macs_G": float(match.group(2)),
            "flops_G": float(match.group(3)),
        }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ Dataset Information
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    dataset_info = {
        "dataset_name": search_value(r"dataset_name=\s*([^\n]+)"),
        "translation_direction": search_value(r"translation_direction=\s*([^\n]+)"),
        "train_samples": search_value(r"train_samples=\s*([\d,]+)", cast=int),
        "test_samples": search_value(r"test_samples=\s*([\d,]+)", cast=int),
    }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ Current Model + Complexity
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # model_name = search_value(r"Model name:\s*([^\n]+)")

    # complexity = search_complexity()

    model_name = search_value(r"Model name:\s*([^\n]+)")

    total_params = search_value(
        r"Total Parameters:\s*([\d,]+)",
        cast=int
    )

    complexity = search_complexity()

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 3️⃣ Optimiser & Activation
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    optimizer_activation = {
        "activation": search_value(r"activation=\s*([^|]+)"),
        "optimizer": search_value(r"optimizer=\s*([^|]+)"),
        "lr": search_value(r"lr=\s*([\d.]+)", cast=float),
        "smoothing": search_value(r"smoothing=\s*([\d.]+)", cast=float),
    }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣ Fairseq Transformer Parameters
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    fairseq_params = {
        "encoder_layers": search_value(r"encoder_layers=\s*(\d+)", cast=int),
        "decoder_layers": search_value(r"decoder_layers=\s*(\d+)", cast=int),
        "encoder_embed_dim": search_value(r"encoder_embed_dim=\s*(\d+)", cast=int),
        "decoder_embed_dim": search_value(r"decoder_embed_dim=\s*(\d+)", cast=int),
        "max_target_positions": search_value(r"max_target_positions=\s*(\d+)", cast=int),
        "dropout": search_value(r"dropout=\s*([\d.]+)", cast=float),
    }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 5️⃣ TDM Parameters only if TDM_Former
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    tdm_params = None

    if model_name == "TDM_Former" or "TDM Parameters" in text:
        tdm_params = {
            "tdm_alpha": search_value(r"tdm_alpha=\s*([\d.]+)", cast=float),
            "tdm_transition_amplification": search_value(r"tdm_transition_amplification=\s*([\d.]+)", cast=float),
            "tdm_num_layers": search_value(r"tdm_num_layers=\s*(\d+)", cast=int),
            "tdm_transition": search_value(r"tdm_transition=\s*(True|False)", cast=bool),
            "tdm_rotation": search_value(r"tdm_rotation=\s*(True|False)", cast=bool),
            "tdm_insertion_type": search_value(r"tdm_insertion_type=\s*([^|]+)"),
            "tdm_scale_init": search_value(r"tdm_scale_init=\s*([\d.]+)", cast=float),
            "tdm_start_epoch": search_value(r"tdm_start_epoch=\s*(\d+)", cast=int),
            "tdm_full_epoch": search_value(r"tdm_full_epoch=\s*(\d+)", cast=int),
            "tdm_gate_type": search_value(r"tdm_gate_type=\s*([^\n|]+)"),
        }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # Final structured output
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # summary = {
    #     "model_name": model_name,
    #     "dataset_info": dataset_info,
    #     "complexity": complexity,
    #     "optimizer_activation": optimizer_activation,
    #     "fairseq_params": fairseq_params,
    #     "tdm_params": tdm_params,
    #     "file_path": file_path,
    # }

    summary = {
        "model_name": model_name,
        "total_params": total_params,
        "dataset_info": dataset_info,
        "complexity": complexity,
        "optimizer_activation": optimizer_activation,
        "fairseq_params": fairseq_params,
        "tdm_params": tdm_params,
        "file_path": file_path,
    }

    return summary
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣ === READ INFERENCE EFFICIENCY RESULTS ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣ === READ INFERENCE EFFICIENCY RESULTS ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────


import re
import os
import pandas as pd


def read_efficiency_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    model_match = re.search(r"Model:\s*(.+)", text)
    model_name = model_match.group(1).strip() if model_match else os.path.basename(log_path)

    times = [float(x) for x in re.findall(r"Inference Time \(s\):\s*([\d.]+)", text)]
    throughputs = [float(x) for x in re.findall(r"Throughput \(words/s\):\s*([\d.]+)", text)]
    memories = [float(x) for x in re.findall(r"Memory \(GB\):\s*([\d.]+)", text)]
    bleus = [float(x) for x in re.findall(r"BLEU:\s*([\d.]+)", text)]

    if len(times) == 0:
        raise ValueError(f"No efficiency entries found in: {log_path}")

    return {
        "Model": model_name,
        "Runs": len(times),
        "BLEU last epoch": sum(bleus) / len(bleus) if bleus else 0.0,
        "Avg Inference Time (s)": sum(times) / len(times),
        "Avg Throughput (word/s)": sum(throughputs) / len(throughputs),
        "Avg Memory (GB)": sum(memories) / len(memories),
    }


def compute_relative_efficiency(baseline_log_path, variant_log_paths, variant_names=None):
    baseline = read_efficiency_log(baseline_log_path)

    if variant_names is None:
        variant_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in variant_log_paths
        ]

    rows = []

    rows.append({
        "Variant": "Baseline",
        "Runs": baseline["Runs"],
        "BLEU last epoch": baseline["BLEU last epoch"],

        "Avg Inference Time (s)": baseline["Avg Inference Time (s)"],
        "Time Δ vs Baseline (%)": 0.0,

        "Avg Throughput (word/s)": baseline["Avg Throughput (word/s)"],
        "Throughput Δ vs Baseline (%)": 0.0,

        "Avg Memory (GB)": baseline["Avg Memory (GB)"],
        "Memory Δ vs Baseline (%)": 0.0,
    })

    for name, path in zip(variant_names, variant_log_paths):
        variant = read_efficiency_log(path)

        rows.append({
            "Variant": name,
            "Runs": variant["Runs"],
            "BLEU last epoch": variant["BLEU last epoch"],

            "Avg Inference Time (s)": variant["Avg Inference Time (s)"],
            "Time Δ vs Baseline (%)": ((variant["Avg Inference Time (s)"] - baseline["Avg Inference Time (s)"]) / baseline["Avg Inference Time (s)"]) * 100,

            "Avg Throughput (word/s)": variant["Avg Throughput (word/s)"],
            "Throughput Δ vs Baseline (%)": ((variant["Avg Throughput (word/s)"] - baseline["Avg Throughput (word/s)"]) / baseline["Avg Throughput (word/s)"]) * 100,

            "Avg Memory (GB)": variant["Avg Memory (GB)"],
            "Memory Δ vs Baseline (%)": ((variant["Avg Memory (GB)"] - baseline["Avg Memory (GB)"]) / baseline["Avg Memory (GB)"]) * 100,
        })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "BLEU last epoch",
        "Avg Inference Time (s)",
        "Time Δ vs Baseline (%)",
        "Avg Throughput (word/s)",
        "Throughput Δ vs Baseline (%)",
        "Avg Memory (GB)",
        "Memory Δ vs Baseline (%)",
    ]

    df[numeric_cols] = df[numeric_cols].round(2)

    return df


def format_efficiency_report(efficiency_df, title="⚡ EFFICIENCY SUMMARY: DE→EN TDM Depth Ablation"):
    rows = []

    for _, row in efficiency_df.iterrows():
        rows.append([
            str(row["Variant"]),
            f"{int(row['Runs'])}",
            f"{row['BLEU last epoch']:.2f}",
            f"{row['Avg Inference Time (s)']:.2f}",
            f"{row['Time Δ vs Baseline (%)']:+.2f}%",
            f"{row['Avg Throughput (word/s)']:.2f}",
            f"{row['Throughput Δ vs Baseline (%)']:+.2f}%",
            f"{row['Avg Memory (GB)']:.2f}",
            f"{row['Memory Δ vs Baseline (%)']:+.2f}%",
        ])

    headers = [
        "Variant",
        "Runs",
        "BLEU last epoch",
        "Inference Time (s)",
        "Time Δ Base",
        "Throughput (word/s)",
        "Thr Δ Base",
        "Mem(GB)",
        "Mem Δ Base",
    ]

    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row_values):
        return " | ".join(
            f"{row_values[i]:<{col_widths[i]}}" if i == 0 else f"{row_values[i]:>{col_widths[i]}}"
            for i in range(len(row_values))
        )

    header_line = format_row(headers)
    middle_line = "─" * len(header_line)

    report_lines = []
    report_lines.append(title)
    report_lines.append(middle_line)
    report_lines.append(header_line)
    report_lines.append(middle_line)

    for row in rows:
        report_lines.append(format_row(row))

    report_lines.append(middle_line)

    return "\n".join(report_lines)


def save_efficiency_report(efficiency_df, txt_path, csv_path):
    efficiency_report_text = format_efficiency_report(efficiency_df)

    print("\n" + efficiency_report_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(efficiency_report_text + "\n")

    efficiency_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved EFFICIENCY summary TXT to: {txt_path}")
    print(f"✅ Saved EFFICIENCY summary CSV to: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 6️⃣ === READ INFERENCE EFFICIENCY RESULTS WITH MEAN ± STD =======================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 6️⃣ === READ INFERENCE EFFICIENCY RESULTS WITH MEAN ± STD =======================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def read_efficiency_log_with_sd(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    times = [float(x) for x in re.findall(r"Inference Time \(s\):\s*([\d.]+)", text)]
    throughputs = [float(x) for x in re.findall(r"Throughput \(words/s\):\s*([\d.]+)", text)]
    memories = [float(x) for x in re.findall(r"Memory \(GB\):\s*([\d.]+)", text)]
    bleus = [float(x) for x in re.findall(r"BLEU:\s*([\d.]+)", text)]

    if len(times) == 0:
        raise ValueError(f"No efficiency entries found in: {log_path}")

    def mean_std(values):
        s = pd.Series(values)
        mean = s.mean()
        std = s.std(ddof=1) if len(values) > 1 else 0.0
        return mean, std

    bleu_mean, bleu_std = mean_std(bleus) if len(bleus) > 0 else (0.0, 0.0)
    time_mean, time_std = mean_std(times)
    throughput_mean, throughput_std = mean_std(throughputs)
    memory_mean, memory_std = mean_std(memories)

    return {
        "Runs": len(times),

        "BLEU last epoch mean": bleu_mean,
        "BLEU last epoch std": bleu_std,

        "Inference Time mean": time_mean,
        "Inference Time std": time_std,

        "Throughput mean": throughput_mean,
        "Throughput std": throughput_std,

        "Memory mean": memory_mean,
        "Memory std": memory_std,
    }


def compute_relative_efficiency_with_sd(baseline_log_path, variant_log_paths, variant_names=None):
    baseline = read_efficiency_log_with_sd(baseline_log_path)

    if variant_names is None:
        variant_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in variant_log_paths
        ]

    rows = []

    rows.append({
        "Variant": "Baseline",
        "Runs": baseline["Runs"],

        "BLEU last epoch": f"{baseline['BLEU last epoch mean']:.2f} ± {baseline['BLEU last epoch std']:.2f}",

        "Inference Time (s)": f"{baseline['Inference Time mean']:.2f} ± {baseline['Inference Time std']:.2f}",
        "Time Δ vs Baseline (%)": "+0.00%",

        "Throughput (word/s)": f"{baseline['Throughput mean']:.2f} ± {baseline['Throughput std']:.2f}",
        "Throughput Δ vs Baseline (%)": "+0.00%",

        "Memory (GB)": f"{baseline['Memory mean']:.2f} ± {baseline['Memory std']:.2f}",
        "Memory Δ vs Baseline (%)": "+0.00%",
    })

    for name, path in zip(variant_names, variant_log_paths):
        variant = read_efficiency_log_with_sd(path)

        time_delta = ((variant["Inference Time mean"] - baseline["Inference Time mean"]) / baseline["Inference Time mean"]) * 100
        throughput_delta = ((variant["Throughput mean"] - baseline["Throughput mean"]) / baseline["Throughput mean"]) * 100
        memory_delta = ((variant["Memory mean"] - baseline["Memory mean"]) / baseline["Memory mean"]) * 100

        rows.append({
            "Variant": name,
            "Runs": variant["Runs"],

            "BLEU last epoch": f"{variant['BLEU last epoch mean']:.2f} ± {variant['BLEU last epoch std']:.2f}",

            "Inference Time (s)": f"{variant['Inference Time mean']:.2f} ± {variant['Inference Time std']:.2f}",
            "Time Δ vs Baseline (%)": f"{time_delta:+.2f}%",

            "Throughput (word/s)": f"{variant['Throughput mean']:.2f} ± {variant['Throughput std']:.2f}",
            "Throughput Δ vs Baseline (%)": f"{throughput_delta:+.2f}%",

            "Memory (GB)": f"{variant['Memory mean']:.2f} ± {variant['Memory std']:.2f}",
            "Memory Δ vs Baseline (%)": f"{memory_delta:+.2f}%",
        })

    return pd.DataFrame(rows)


def format_efficiency_report_with_sd(efficiency_df, title="⚡ EFFICIENCY SUMMARY WITH STD: DE→EN TDM Depth Ablation"):

    headers = list(efficiency_df.columns)
    rows = efficiency_df.astype(str).values.tolist()

    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row_values):
        return " | ".join(
            f"{row_values[i]:<{col_widths[i]}}" if i == 0 else f"{row_values[i]:>{col_widths[i]}}"
            for i in range(len(row_values))
        )

    header_line = format_row(headers)
    middle_line = "─" * len(header_line)

    report_lines = []
    report_lines.append(title)
    report_lines.append(middle_line)
    report_lines.append(header_line)
    report_lines.append(middle_line)

    for row in rows:
        report_lines.append(format_row(row))

    report_lines.append(middle_line)

    return "\n".join(report_lines)


def save_efficiency_report_with_sd(efficiency_df, txt_path, csv_path):
    efficiency_report_text = format_efficiency_report_with_sd(efficiency_df)

    print("\n" + efficiency_report_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(efficiency_report_text + "\n")

    efficiency_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved EFFICIENCY summary with STD TXT to: {txt_path}")
    print(f"✅ Saved EFFICIENCY summary with STD CSV to: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 7️⃣ === EFFICIENCY TABLE WITH BLEU LAST EPOCH + BLEU BEST EPOCH + MEAN ± STD ====================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 7️⃣ === EFFICIENCY TABLE WITH BLEU LAST EPOCH + BLEU BEST EPOCH + MEAN ± STD ====================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def compute_relative_efficiency_with_sd_and_best_bleu(
    baseline_log_path,
    variant_log_paths,
    variant_names=None,
    best_bleu_by_variant=None
):
    baseline = read_efficiency_log_with_sd(baseline_log_path)

    if variant_names is None:
        variant_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in variant_log_paths
        ]

    if best_bleu_by_variant is None:
        best_bleu_by_variant = {}

    rows = []

    rows.append({
        "Variant": "Baseline",
        "Runs": baseline["Runs"],

        "BLEU last epoch": f"{baseline['BLEU last epoch mean']:.2f} ± {baseline['BLEU last epoch std']:.2f}",
        "BLEU best epoch": f"{best_bleu_by_variant.get('Baseline', 0.0):.2f}",

        "Inference Time (s)": f"{baseline['Inference Time mean']:.2f} ± {baseline['Inference Time std']:.2f}",
        "Time Δ vs Baseline (%)": "+0.00%",

        "Throughput (word/s)": f"{baseline['Throughput mean']:.2f} ± {baseline['Throughput std']:.2f}",
        "Throughput Δ vs Baseline (%)": "+0.00%",

        "Memory (GB)": f"{baseline['Memory mean']:.2f} ± {baseline['Memory std']:.2f}",
        "Memory Δ vs Baseline (%)": "+0.00%",
    })

    for name, path in zip(variant_names, variant_log_paths):
        variant = read_efficiency_log_with_sd(path)

        time_delta = ((variant["Inference Time mean"] - baseline["Inference Time mean"]) / baseline["Inference Time mean"]) * 100
        throughput_delta = ((variant["Throughput mean"] - baseline["Throughput mean"]) / baseline["Throughput mean"]) * 100
        memory_delta = ((variant["Memory mean"] - baseline["Memory mean"]) / baseline["Memory mean"]) * 100

        rows.append({
            "Variant": name,
            "Runs": variant["Runs"],

            "BLEU last epoch": f"{variant['BLEU last epoch mean']:.2f} ± {variant['BLEU last epoch std']:.2f}",
            "BLEU best epoch": f"{best_bleu_by_variant.get(name, 0.0):.2f}",

            "Inference Time (s)": f"{variant['Inference Time mean']:.2f} ± {variant['Inference Time std']:.2f}",
            "Time Δ vs Baseline (%)": f"{time_delta:+.2f}%",

            "Throughput (word/s)": f"{variant['Throughput mean']:.2f} ± {variant['Throughput std']:.2f}",
            "Throughput Δ vs Baseline (%)": f"{throughput_delta:+.2f}%",

            "Memory (GB)": f"{variant['Memory mean']:.2f} ± {variant['Memory std']:.2f}",
            "Memory Δ vs Baseline (%)": f"{memory_delta:+.2f}%",
        })

    return pd.DataFrame(rows)


def format_efficiency_report_with_sd_and_best_bleu(
    efficiency_df,
    title="⚡ EFFICIENCY SUMMARY WITH STD: DE→EN TDM Depth Ablation"
):
    headers = list(efficiency_df.columns)
    rows = efficiency_df.astype(str).values.tolist()

    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row_values):
        return " | ".join(
            f"{row_values[i]:<{col_widths[i]}}" if i == 0 else f"{row_values[i]:>{col_widths[i]}}"
            for i in range(len(row_values))
        )

    header_line = format_row(headers)
    middle_line = "─" * len(header_line)

    report_lines = []
    report_lines.append(title)
    report_lines.append(middle_line)
    report_lines.append(header_line)
    report_lines.append(middle_line)

    for row in rows:
        report_lines.append(format_row(row))

    report_lines.append(middle_line)

    return "\n".join(report_lines)


def save_efficiency_report_with_sd_and_best_bleu(efficiency_df, txt_path, csv_path):
    efficiency_report_text = format_efficiency_report_with_sd_and_best_bleu(efficiency_df)

    print("\n" + efficiency_report_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(efficiency_report_text + "\n")

    efficiency_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved EFFICIENCY summary with STD + Best BLEU TXT to: {txt_path}")
    print(f"✅ Saved EFFICIENCY summary with STD + Best BLEU CSV to: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 7. DEFINE FILE PATH | XXX --------------------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 7. DEFINE FILE PATH | XXX --------------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== 7️⃣.1️⃣ Define Train/Test/Efficiency log file paths ================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def build_log_path(curve_id, folder_name):
    # plot_name    = getattr(exp_args, f"Plot{curve_id}")
    mode_name    = getattr(exp_args, f"mode_name_Plot{curve_id}")
    net_name     = getattr(exp_args, f"net_Plot{curve_id}")
    dataset_name = getattr(exp_args, f"dataset_name_Plot{curve_id}")

    if exp_args.evaluation_mode == "test":
        result_folder = "Results"
        file_name = f"TestCheckpoints_{net_name}_{dataset_name}_{exp_args.optimizer1}_{mode_name}.txt"

    elif exp_args.evaluation_mode == "train":
        result_folder = "Results"
        file_name = f"Train_{net_name}_{dataset_name}_{exp_args.optimizer1}_{mode_name}.txt"

    elif exp_args.evaluation_mode == "efficiency":
        result_folder = "Results_efficiency_test"
        file_name = f"{net_name}_{dataset_name}_{exp_args.optimizer1}_{mode_name}_efficiency_test.txt"

    else:
        raise ValueError(f"Unknown Evaluation mode: {exp_args.evaluation_mode}")

    return os.path.join(
        DATA_TEST_PATH,
        folder_name,
        result_folder,
        dataset_name,
        net_name,
        file_name
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define all log file paths ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────

Path_Plot_1 = build_log_path("1", "1.1-Transformer_Baselines_DE_EN-Baseline")  # Baseline
Path_Plot_2 = build_log_path("2", "1.2-Transformer_Baselines_DE_EN-L6")        # L6
Path_Plot_3 = build_log_path("3", "1.3-Transformer_Baselines_DE_EN-L5_L6")     # L5-L6
Path_Plot_4 = build_log_path("4", "1.4-Transformer_Baselines_DE_EN-L4_L6")     # L4-L6
Path_Plot_5 = build_log_path("5", "1.5-Transformer_Baselines_DE_EN-L3_L6")     # L3-L6

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print log file paths sanity check ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────

print(f"\n📁 {exp_args.evaluation_mode.upper()} log file paths:")
print("─" * 100)

all_paths = {
    "DE→EN | Baseline":       Path_Plot_1,
    "DE→EN | TDM L6":         Path_Plot_2,
    "DE→EN | TDM L5-L6":      Path_Plot_3,
    "DE→EN | TDM L4-L6":      Path_Plot_4,
    "DE→EN | TDM L3-L6":      Path_Plot_5,
}

for name, path in all_paths.items():
    print(f"🧪 {name}:")
    print(f"   {path}\n")

print("─" * 100)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# %%

########################################################################################################################
####-------| NOTE 8. CALL EFFICIENCY TABLE + SAVE TXT/CSV  | XXX -----------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 8. CALL EFFICIENCY TABLE + SAVE TXT/CSV  | XXX -----------------------------------####################
########################################################################################################################



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚡1️⃣ === CALL EFFICIENCY TABLE + SAVE TXT/CSV ==================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

variant_paths = [
    Path_Plot_2,  # TDM L6
    Path_Plot_3,  # TDM L5-L6
    Path_Plot_4,  # TDM L4-L6
    Path_Plot_5,  # TDM L3-L6
]

variant_names = [
    "TDM L6",
    "TDM L5-L6",
    "TDM L4-L6",
    "TDM L3-L6",
]

efficiency_df = compute_relative_efficiency(
    baseline_log_path=Path_Plot_1,
    variant_log_paths=variant_paths,
    variant_names=variant_names
)

efficiency_txt_path = os.path.join(
    DATA_TEST_PATH,
    "TDM_Former_DE_EN_Depth_Ablation_EFFICIENCY_Summary.txt"
)

efficiency_csv_path = os.path.join(
    DATA_TEST_PATH,
    "TDM_Former_DE_EN_Depth_Ablation_EFFICIENCY_Summary.csv"
)

save_efficiency_report(
    efficiency_df=efficiency_df,
    txt_path=efficiency_txt_path,
    csv_path=efficiency_csv_path
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣⚡ === CALL EFFICIENCY TABLE WITH MEAN ± STD + SAVE TXT/CSV ==================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

variant_paths = [
    Path_Plot_2,  # TDM L6
    Path_Plot_3,  # TDM L5-L6
    Path_Plot_4,  # TDM L4-L6
    Path_Plot_5,  # TDM L3-L6
]

variant_names = [
    "TDM L6",
    "TDM L5-L6",
    "TDM L4-L6",
    "TDM L3-L6",
]

efficiency_df_with_sd = compute_relative_efficiency_with_sd(
    baseline_log_path=Path_Plot_1,
    variant_log_paths=variant_paths,
    variant_names=variant_names
)

efficiency_txt_path_with_sd = os.path.join(
    DATA_TEST_PATH,
    "TDM_Former_DE_EN_Depth_Ablation_EFFICIENCY_Summary_WITH_STD.txt"
)

efficiency_csv_path_with_sd = os.path.join(
    DATA_TEST_PATH,
    "TDM_Former_DE_EN_Depth_Ablation_EFFICIENCY_Summary_WITH_STD.csv"
)

save_efficiency_report_with_sd(
    efficiency_df=efficiency_df_with_sd,
    txt_path=efficiency_txt_path_with_sd,
    csv_path=efficiency_csv_path_with_sd
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣⚡ === CALL EFFICIENCY TABLE WITH BLEU LAST + BEST + MEAN ± STD ================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

variant_paths = [
    Path_Plot_2,  # TDM L6
    Path_Plot_3,  # TDM L5-L6
    Path_Plot_4,  # TDM L4-L6
    Path_Plot_5,  # TDM L3-L6
]

variant_names = [
    "TDM L6",
    "TDM L5-L6",
    "TDM L4-L6",
    "TDM L3-L6",
]

best_bleu_by_variant = {
    "Baseline": 30.81,
    "TDM L6": 31.16,
    "TDM L5-L6": 31.37,
    "TDM L4-L6": 31.51,
    "TDM L3-L6": 31.54,
}

efficiency_df_with_sd_best_bleu = compute_relative_efficiency_with_sd_and_best_bleu(
    baseline_log_path=Path_Plot_1,
    variant_log_paths=variant_paths,
    variant_names=variant_names,
    best_bleu_by_variant=best_bleu_by_variant
)

efficiency_txt_path_with_sd_best_bleu = os.path.join(
    DATA_TEST_PATH,
    "TDM_Former_DE_EN_Depth_Ablation_EFFICIENCY_Summary_WITH_STD_AND_BEST_BLEU.txt"
)

efficiency_csv_path_with_sd_best_bleu = os.path.join(
    DATA_TEST_PATH,
    "TDM_Former_DE_EN_Depth_Ablation_EFFICIENCY_Summary_WITH_STD_AND_BEST_BLEU.csv"
)

save_efficiency_report_with_sd_and_best_bleu(
    efficiency_df=efficiency_df_with_sd_best_bleu,
    txt_path=efficiency_txt_path_with_sd_best_bleu,
    csv_path=efficiency_csv_path_with_sd_best_bleu
)

# ─────────────────────────────────────────────────────────────────────────────────────────────────





# %%





########################################################################################################################
####-------| NOTE 9.1.A TDM-FORMER INFERENCE EFFICIENCY BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN ---- #############
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.1.A TDM-FORMER INFERENCE EFFICIENCY BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN ---- #############
########################################################################################################################

"""
X-axis = Inference-time
Y-axis = BLEU Gain
Bubble size = Throughput Drop
Text label = Throughput Drop
"""


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, Inference Time(s), Throughput(words/s), Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Efficiency Data
    ("Baseline",        30.81, 6.03, 5204.05, "--"),
    ("TDM L6",          31.16, 6.40, 4985.42, "L6"),
    ("TDM L5-L6",       31.37, 6.54, 4870.09, "L5-L6"),
    ("TDM L4-L6",       31.51, 6.80, 4660.17, "L4-L6"),
    ("TDM L3-L6",       31.54, 7.11, 4485.32, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Depth Efficiency ===
COLORS = {
    "Baseline_model":        "#2E2E2E",    # Baseline Transformer
    "TDM_L6_model":          "#EF476F",    # TDM L6
    "TDM_L5_L6_model":       "#06D6A0",    # TDM L5-L6
    "TDM_L4_L6_model":       "#E49B0F",    # TDM L4-L6
    "TDM_L3_L6_model":       "#8338EC",    # TDM L3-L6
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 5️⃣🔗================ GENERATE PLOTS 🔑=====================🔗
# ===============================================================   

def plot_models_comparison(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "Baseline":    "Baseline_model",
        "TDM L6":      "TDM_L6_model",
        "TDM L5-L6":   "TDM_L5_L6_model",
        "TDM L4-L6":   "TDM_L4_L6_model",
        "TDM L3-L6":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === PER-BUBBLE OFFSETS ===
    # ─────────────────────────────────────────────
    ANNOT_OFFSET = {
        ("ALL", "Baseline"):   (-0.06, -0.136),
        ("ALL", "TDM L6"):     (-0.06, -0.123),
        ("ALL", "TDM L5-L6"):  (-0.05, -0.116),
        ("ALL", "TDM L4-L6"):  (-0.055,  -0.104),
        ("ALL", "TDM L3-L6"):  (-0.05, -0.088),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 20,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === BUBBLE SIZE = THROUGHPUT ===
    # Higher throughput = larger bubble
    # ─────────────────────────────────────────────
    throughputs = [thr for _, _, _, thr, _ in data]
    min_thr = min(throughputs)
    max_thr = max(throughputs)

    def scale_bubble(thr):
        return 450 + ((thr - min_thr) / (max_thr - min_thr)) * 1050

    # ─────────────────────────────────────────────
    # ⚙️ === DRAW BUBBLES ===
    # X = Inference Time, Y = BLEU Best, Bubble = Throughput
    # ─────────────────────────────────────────────
    for model, bleu_best, inference_time, throughput, layers in data:

        color_key = MODEL2COLOR[model]
        bubble_size = scale_bubble(throughput)

        ax.scatter(
            [inference_time], [bleu_best],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM L4-L6" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === ANNOTATE THROUGHPUT ===
    # ─────────────────────────────────────────────
    for model, bleu_best, inference_time, throughput, layers in data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            inference_time + dx,
            bleu_best + dy,
            rf"\textbf{{{throughput:.0f}}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Inference Time (s)}")
    # ax.set_ylabel(r"\textbf{BLEU Last}")
    ax.set_ylabel(r"\textbf{BLEU}")
    ax.grid(True, linestyle="--", alpha=0.35)

    # ax.set_xlim(5.85, 7.30)
    ax.set_xlim(5.90, 7.30)
    ax.set_xticks([6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2])

    ax.set_ylim(30.35, 31.65)
    ax.set_yticks([30.4, 30.6, 30.8, 31.0, 31.2, 31.4, 31.6])

    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="lower right",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_InferenceTimeX_BLEUBestY_ThroughputBubble_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_InferenceTimeX_BLEUBestY_ThroughputBubble_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOTS 🔑=======================🔗
# ===============================================================   
plot_models_comparison()
# ────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 9.1.B TDM-FORMER INFERENCE EFFICIENCY BUBBLE PLOT WITH DASHED LINE | XXX -------------------##########
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.1.B TDM-FORMER INFERENCE EFFICIENCY BUBBLE PLOT WITH DASHED LINE | XXX -------------------##########
########################################################################################################################

"""
X-axis = Inference Time (s)
Y-axis = BLEU Best
Bubble size = Throughput (words/s)
Text label = Throughput (words/s)
Dashed line = connects variants in increasing TDM depth order
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, Inference Time(s), Throughput(words/s), Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Efficiency Data
    ("Baseline",          30.81, 6.03, 5204.05, "--"),
    ("TDM (L6)",          31.16, 6.40, 4985.42, "L6"),
    ("TDM (L5-L6)",       31.37, 6.54, 4870.09, "L5-L6"),
    ("TDM (L4-L6)",       31.51, 6.80, 4660.17, "L4-L6"),
    ("TDM (L3-L6)",       31.54, 7.11, 4485.32, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Depth Efficiency ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline_model":        "#2E2E2E",    # Baseline Transformer
    "TDM_L6_model":          "#EF476F",    # TDM L6
    "TDM_L5_L6_model":       "#06D6A0",    # TDM L5-L6
    "TDM_L4_L6_model":       "#E49B0F",    # TDM L4-L6
    "TDM_L3_L6_model":       "#8338EC",    # TDM L3-L6
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ===============================================================
# 5️⃣🔗================ GENERATE PLOTS 🔑=====================🔗
# ===============================================================
def plot_models_comparison_dashed(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "Baseline":      "Baseline_model",
        "TDM (L6)":      "TDM_L6_model",
        "TDM (L5-L6)":   "TDM_L5_L6_model",
        "TDM (L4-L6)":   "TDM_L4_L6_model",
        "TDM (L3-L6)":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === PER-BUBBLE OFFSETS ===
    # ─────────────────────────────────────────────
    ANNOT_OFFSET = {
        ("ALL", "Baseline"):     (-0.06, -0.136),
        ("ALL", "TDM (L6)"):     (-0.06, -0.123),
        ("ALL", "TDM (L5-L6)"):  (-0.05, -0.116),
        ("ALL", "TDM (L4-L6)"):  (-0.055, -0.104),
        ("ALL", "TDM (L3-L6)"):  (-0.05, -0.088),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 20,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === BUBBLE SIZE = THROUGHPUT ===
    # Higher throughput = larger bubble
    # ─────────────────────────────────────────────
    throughputs = [thr for _, _, _, thr, _ in data]
    min_thr = min(throughputs)
    max_thr = max(throughputs)

    def scale_bubble(thr):
        if max_thr == min_thr:
            return 700
        return 450 + ((thr - min_thr) / (max_thr - min_thr)) * 1050

    # ─────────────────────────────────────────────
    # 📈 === DASHED LINE: connect points in model order
    # ─────────────────────────────────────────────
    x_line = [inference_time for _, _, inference_time, _, _ in data]
    y_line = [bleu_best for _, bleu_best, _, _, _ in data]

    # ax.plot(
    #     x_line,
    #     y_line,
    #     linestyle="--",
    #     linewidth=1.2,
    #     color="#A9A9A9",
    #     alpha=0.9,
    #     zorder=1
    # )

    trend_line, = ax.plot(
        x_line,
        y_line,
        linestyle=(0, (1, 4)),   # round dotted line
        linewidth=1.6,
        color="#5A5A5A",
        alpha=0.85,
        zorder=1
    )

    trend_line.set_dash_capstyle("round")
    # ─────────────────────────────────────────────
    # ⚙️ === DRAW BUBBLES ===
    # X = Inference Time, Y = BLEU Best, Bubble = Throughput
    # ─────────────────────────────────────────────
    for model, bleu_best, inference_time, throughput, layers in data:

        color_key = MODEL2COLOR[model]
        bubble_size = scale_bubble(throughput)

        ax.scatter(
            [inference_time], [bleu_best],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM (L4-L6)" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === ANNOTATE THROUGHPUT ===
    # ─────────────────────────────────────────────
    for model, bleu_best, inference_time, throughput, layers in data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            inference_time + dx,
            bleu_best + dy,
            rf"\textbf{{{throughput:.0f}}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Inference Time (s)}")
    ax.set_ylabel(r"\textbf{BLEU}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(5.90, 7.30)
    ax.set_xticks([6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2])

    ax.set_ylim(30.35, 31.65)
    ax.set_yticks([30.4, 30.6, 30.8, 31.0, 31.2, 31.4, 31.6])

    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="lower right",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_InferenceTimeX_BLEUBestY_ThroughputBubble_DashedLine_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_InferenceTimeX_BLEUBestY_ThroughputBubble_DashedLine_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOTS 🔑=======================🔗
# ===============================================================
plot_models_comparison_dashed()
# ────────────────────────────────────────────────────────────────









########################################################################################################################
####-------| NOTE 9.2 TDM-FORMER TRADE-OFF BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN ---- ##########################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.2 TDM-FORMER TRADE-OFF BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN ---- ##########################
########################################################################################################################

"""
X-axis = Throughput Drop vs Baseline (%)
Y-axis = BLEU Gain vs Baseline
Bubble size = Inference-time overhead (%)
Text label = Inference-time overhead (%)
"""


import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, BLEU Gain, Time Overhead %, Throughput Drop %, Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Trade-Off Data
    ("Baseline",        30.81, 0.00,  0.00,  0.00, "--"),
    ("TDM L6",          31.16, 0.35,  6.05,  4.20, "L6"),
    ("TDM L5-L6",       31.37, 0.56,  8.37,  6.42, "L5-L6"),
    ("TDM L4-L6",       31.51, 0.70, 12.70, 10.45, "L4-L6"),
    ("TDM L3-L6",       31.54, 0.73, 17.84, 13.81, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Trade-Off ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline_model":        "#2E2E2E",
    "TDM_L6_model":          "#EF476F",
    "TDM_L5_L6_model":       "#06D6A0",
    "TDM_L4_L6_model":       "#E49B0F",
    "TDM_L3_L6_model":       "#8338EC",
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ===============================================================
# 5️⃣🔗================ GENERATE PLOT 🔑======================🔗
# ===============================================================
def plot_tdm_tradeoff_oneplot(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    MODEL2COLOR = {
        "Baseline":    "Baseline_model",
        "TDM L6":      "TDM_L6_model",
        "TDM L5-L6":   "TDM_L5_L6_model",
        "TDM L4-L6":   "TDM_L4_L6_model",
        "TDM L3-L6":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === Plot 1 data
    # X-axis = Throughput Drop vs Baseline (%)
    # Y-axis = BLEU Gain vs Baseline
    # Bubble size = Inference-time overhead (%)
    # Text label = Inference-time overhead (%)
    # ─────────────────────────────────────────────
    tradeoff_data = []

    for model, bleu_best, bleu_gain, time_overhead, throughput_drop, layers in data:
        tradeoff_data.append((model, bleu_gain, time_overhead, throughput_drop, layers))

    ANNOT_OFFSET = {
        ("ALL", "Baseline"):   (-0.7,  -0.053),
        ("ALL", "TDM L6"):     (-0.7,  -0.08),
        ("ALL", "TDM L5-L6"):  (-0.7, -0.085),
        ("ALL", "TDM L4-L6"):  (-0.8,  -0.093),
        ("ALL", "TDM L3-L6"):  (-0.8, -0.103),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 15,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === Bubble size = inference-time overhead (%)
    # ─────────────────────────────────────────────
    overheads = [time_overhead for _, _, time_overhead, _, _ in tradeoff_data]
    min_ov = min(overheads)
    max_ov = max(overheads)

    def scale_bubble(time_overhead):
        if max_ov == min_ov:
            return 500
        return 250 + ((time_overhead - min_ov) / (max_ov - min_ov)) * 1200

    # ─────────────────────────────────────────────
    # ⚙️ === Draw bubbles
    # ─────────────────────────────────────────────
    for model, bleu_gain, time_overhead, throughput_drop, layers in tradeoff_data:

        color_key = MODEL2COLOR[model]
        bubble_size = 160 if model == "Baseline" else scale_bubble(time_overhead)

        ax.scatter(
            [throughput_drop], [bleu_gain],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM L4-L6" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === Annotate inference-time overhead (%)
    # ─────────────────────────────────────────────
    for model, bleu_gain, time_overhead, throughput_drop, layers in tradeoff_data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            throughput_drop + dx,
            bleu_gain + dy,
            rf"\textbf{{{time_overhead:.2f}\%}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === Labels / axis
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Throughput Drop vs Baseline (\%)}")
    ax.set_ylabel(r"\textbf{BLEU Gain vs Baseline}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1.0, 15.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

    ax.set_ylim(-0.10, 0.85)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

    # ─────────────────────────────────────────────
    # 🔍 === Legend
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in tradeoff_data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === Save
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_ThroughputDropX_BLEUGainY_TimeOverheadBubble_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_ThroughputDropX_BLEUGainY_TimeOverheadBubble_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOT 🔑========================🔗
# ===============================================================
plot_tdm_tradeoff_oneplot()
# ────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 9.3 TDM-FORMER TRADE-OFF BUBBLE PLOT 2 | XXX ----🎀📣 Y-axis: BLEU GAIN ---- ########################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.3 TDM-FORMER TRADE-OFF BUBBLE PLOT 2 | XXX ----🎀📣 Y-axis: BLEU GAIN ---- ########################
########################################################################################################################

"""
X-axis = Inference-time overhead (%)
Y-axis = BLEU Gain vs Baseline
Bubble size = Throughput Drop vs Baseline (%)
Text label = Throughput Drop vs Baseline (%)
"""


import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, BLEU Gain, Time Overhead %, Throughput Drop %, Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Trade-Off Data
    ("Baseline",        30.81, 0.00,  0.00,  0.00, "--"),
    ("TDM L6",          31.16, 0.35,  6.05,  4.20, "L6"),
    ("TDM L5-L6",       31.37, 0.56,  8.37,  6.42, "L5-L6"),
    ("TDM L4-L6",       31.51, 0.70, 12.70, 10.45, "L4-L6"),
    ("TDM L3-L6",       31.54, 0.73, 17.84, 13.81, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Trade-Off ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline_model":        "#2E2E2E",
    "TDM_L6_model":          "#EF476F",
    "TDM_L5_L6_model":       "#06D6A0",
    "TDM_L4_L6_model":       "#E49B0F",
    "TDM_L3_L6_model":       "#8338EC",
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ===============================================================
# 5️⃣🔗================ GENERATE PLOT 2 🔑====================🔗
# ===============================================================
def plot_tdm_tradeoff_oneplot_2(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    MODEL2COLOR = {
        "Baseline":    "Baseline_model",
        "TDM L6":      "TDM_L6_model",
        "TDM L5-L6":   "TDM_L5_L6_model",
        "TDM L4-L6":   "TDM_L4_L6_model",
        "TDM L3-L6":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === Plot 2 data
    # X-axis = Inference-time overhead (%)
    # Y-axis = BLEU Gain vs Baseline
    # Bubble size = Throughput Drop vs Baseline (%)
    # Text label = Throughput Drop vs Baseline (%)
    # ─────────────────────────────────────────────
    tradeoff_data = []

    for model, bleu_best, bleu_gain, time_overhead, throughput_drop, layers in data:
        tradeoff_data.append((model, bleu_gain, time_overhead, throughput_drop, layers))

    ANNOT_OFFSET = {
        ("ALL", "Baseline"):   (-0.8,  -0.053),
        ("ALL", "TDM L6"):     (-0.9,  -0.08),
        ("ALL", "TDM L5-L6"):  (-0.9, -0.085),
        ("ALL", "TDM L4-L6"):  (-1.0,  -0.093),
        ("ALL", "TDM L3-L6"):  (-1.1, -0.103),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 15,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === Bubble size = throughput drop vs baseline (%)
    # ─────────────────────────────────────────────
    throughput_drops = [throughput_drop for _, _, _, throughput_drop, _ in tradeoff_data]
    min_drop = min(throughput_drops)
    max_drop = max(throughput_drops)

    def scale_bubble(throughput_drop):
        if max_drop == min_drop:
            return 500
        return 250 + ((throughput_drop - min_drop) / (max_drop - min_drop)) * 1200

    # ─────────────────────────────────────────────
    # ⚙️ === Draw bubbles
    # ─────────────────────────────────────────────
    for model, bleu_gain, time_overhead, throughput_drop, layers in tradeoff_data:

        color_key = MODEL2COLOR[model]
        bubble_size = 160 if model == "Baseline" else scale_bubble(throughput_drop)

        ax.scatter(
            [time_overhead], [bleu_gain],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM L4-L6" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === Annotate throughput drop vs baseline (%)
    # ─────────────────────────────────────────────
    for model, bleu_gain, time_overhead, throughput_drop, layers in tradeoff_data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            time_overhead + dx,
            bleu_gain + dy,
            rf"\textbf{{{throughput_drop:.2f}\%}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === Labels / axis
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Inference-Time Overhead vs Baseline (\%)}")
    ax.set_ylabel(r"\textbf{BLEU Gain vs Baseline}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1.0, 19.5)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18])

    ax.set_ylim(-0.10, 0.85)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

    # ─────────────────────────────────────────────
    # 🔍 === Legend
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in tradeoff_data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === Save
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_TimeOverheadX_BLEUGainY_ThroughputDropBubble_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_TimeOverheadX_BLEUGainY_ThroughputDropBubble_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOT 2 🔑======================🔗
# ===============================================================
plot_tdm_tradeoff_oneplot_2()
# ────────────────────────────────────────────────────────────────








########################################################################################################################
####-------| NOTE 9.4.A TDM-FORMER TRADE-OFF BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN (%) ---- ####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.4.A TDM-FORMER TRADE-OFF BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN (%) ---- ####################
########################################################################################################################

"""
X-axis = Throughput Drop vs Baseline (%)
Y-axis = BLEU Gain vs Baseline (%)
Bubble size = Inference-time overhead vs Baseline  (%)
Text label = Inference-time overhead vs Baseline  (%)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, BLEU Gain, BLEU Gain %, Time Overhead %, Throughput Drop %, Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Trade-Off Data
    ("Baseline",        30.81, 0.00, 0.00,  0.00,  0.00, "--"),
    ("TDM L6",          31.16, 0.35, 1.14,  6.05,  4.20, "L6"),
    ("TDM L5-L6",       31.37, 0.56, 1.82,  8.37,  6.42, "L5-L6"),
    ("TDM L4-L6",       31.51, 0.70, 2.27, 12.70, 10.45, "L4-L6"),
    ("TDM L3-L6",       31.54, 0.73, 2.37, 17.84, 13.81, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Trade-Off ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline_model":        "#2E2E2E",
    "TDM_L6_model":          "#EF476F",
    "TDM_L5_L6_model":       "#06D6A0",
    "TDM_L4_L6_model":       "#E49B0F",
    "TDM_L3_L6_model":       "#8338EC",
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ===============================================================
# 5️⃣🔗================ GENERATE PLOT 🔑======================🔗
# ===============================================================
def plot_tdm_tradeoff_oneplot(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    MODEL2COLOR = {
        "Baseline":    "Baseline_model",
        "TDM L6":      "TDM_L6_model",
        "TDM L5-L6":   "TDM_L5_L6_model",
        "TDM L4-L6":   "TDM_L4_L6_model",
        "TDM L3-L6":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === Plot 1 data
    # X-axis = Throughput Drop vs Baseline (%)
    # Y-axis = BLEU Gain vs Baseline (%)
    # Bubble size = Inference-time overhead (%)
    # Text label = Inference-time overhead (%)
    # ─────────────────────────────────────────────
    tradeoff_data = []

    for model, bleu_best, bleu_gain, bleu_gain_pct, time_overhead, throughput_drop, layers in data:
        tradeoff_data.append((model, bleu_gain_pct, time_overhead, throughput_drop, layers))

    ANNOT_OFFSET = {
        ("ALL", "Baseline"):   (-0.75,  0.12),
        ("ALL", "TDM L6"):     (-0.70,  -0.22),
        ("ALL", "TDM L5-L6"):  (-0.70, -0.24),
        ("ALL", "TDM L4-L6"):  (-1.00,  -0.265),
        ("ALL", "TDM L3-L6"):  (-1.00, -0.295),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 15,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === Bubble size = inference-time overhead (%)
    # ─────────────────────────────────────────────
    overheads = [time_overhead for _, _, time_overhead, _, _ in tradeoff_data]
    min_ov = min(overheads)
    max_ov = max(overheads)

    def scale_bubble(time_overhead):
        if max_ov == min_ov:
            return 500
        return 250 + ((time_overhead - min_ov) / (max_ov - min_ov)) * 1200

    # ─────────────────────────────────────────────
    # ⚙️ === Draw bubbles
    # ─────────────────────────────────────────────
    for model, bleu_gain_pct, time_overhead, throughput_drop, layers in tradeoff_data:

        color_key = MODEL2COLOR[model]
        bubble_size = 160 if model == "Baseline" else scale_bubble(time_overhead)

        ax.scatter(
            [throughput_drop], [bleu_gain_pct],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM L4-L6" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === Annotate inference-time overhead (%)
    # ─────────────────────────────────────────────
    for model, bleu_gain_pct, time_overhead, throughput_drop, layers in tradeoff_data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            throughput_drop + dx,
            bleu_gain_pct + dy,
            rf"\textbf{{{time_overhead:.2f}\%}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === Labels / axis
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Throughput Drop vs Baseline (\%)}")
    ax.set_ylabel(r"\textbf{BLEU Gain vs Baseline (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1.5, 15.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

    ax.set_ylim(-0.15, 2.65)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # ─────────────────────────────────────────────
    # 🔍 === Legend
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in tradeoff_data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === Save
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_ThroughputDropX_BLEUGainPctY_TimeOverheadBubble_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_ThroughputDropX_BLEUGainPctY_TimeOverheadBubble_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOT 🔑========================🔗
# ===============================================================
plot_tdm_tradeoff_oneplot()
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 9.4.B TDM-FORMER TRADE-OFF BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN (%) ---- ####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.4.B TDM-FORMER TRADE-OFF BUBBLE PLOT | XXX ----🎀📣 Y-axis: BLEU GAIN (%) ---- ####################
########################################################################################################################

"""
X-axis = Throughput Drop vs Baseline (%)
Y-axis = BLEU Gain vs Baseline (%)
Bubble size = Inference-time overhead vs Baseline (%)
Text label = Inference-time overhead vs Baseline (%)
Trend line = thin dashed line connecting variants in depth order
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, BLEU Gain, BLEU Gain %, Time Overhead %, Throughput Drop %, Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Trade-Off Data
    ("Baseline",          30.81, 0.00, 0.00,  0.00,  0.00, "--"),
    ("TDM (L6)",          31.16, 0.35, 1.14,  6.05,  4.20, "L6"),
    ("TDM (L5-L6)",       31.37, 0.56, 1.82,  8.37,  6.42, "L5-L6"),
    ("TDM (L4-L6)",       31.51, 0.70, 2.27, 12.70, 10.45, "L4-L6"),
    ("TDM (L3-L6)",       31.54, 0.73, 2.37, 17.84, 13.81, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Trade-Off ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline_model":        "#2E2E2E",
    "TDM_L6_model":          "#EF476F",
    "TDM_L5_L6_model":       "#06D6A0",
    "TDM_L4_L6_model":       "#E49B0F",
    "TDM_L3_L6_model":       "#8338EC",
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ===============================================================
# 5️⃣🔗================ GENERATE PLOT 🔑======================🔗
# ===============================================================
def plot_tdm_tradeoff_oneplot_with_line(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    MODEL2COLOR = {
        "Baseline":      "Baseline_model",
        "TDM (L6)":      "TDM_L6_model",
        "TDM (L5-L6)":   "TDM_L5_L6_model",
        "TDM (L4-L6)":   "TDM_L4_L6_model",
        "TDM (L3-L6)":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === Plot data
    # X-axis = Throughput Drop vs Baseline (%)
    # Y-axis = BLEU Gain vs Baseline (%)
    # Bubble size = Inference-time overhead vs Baseline (%)
    # Text label = Inference-time overhead vs Baseline (%)
    # ─────────────────────────────────────────────
    tradeoff_data = []

    for model, bleu_best, bleu_gain, bleu_gain_pct, time_overhead, throughput_drop, layers in data:
        tradeoff_data.append((model, bleu_gain_pct, time_overhead, throughput_drop, layers))

    ANNOT_OFFSET = {
        ("ALL", "Baseline"):   (0.41,  -0.014),
        ("ALL", "TDM (L6)"):     (-0.5, -0.22),
        ("ALL", "TDM (L5-L6)"):  (-0.5, -0.24),
        ("ALL", "TDM (L4-L6)"):  (-1.00, -0.265),
        ("ALL", "TDM (L3-L6)"):  (-1.00, -0.295),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 15,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === Bubble size = inference-time overhead vs baseline (%)
    # ─────────────────────────────────────────────
    overheads = [time_overhead for _, _, time_overhead, _, _ in tradeoff_data]
    min_ov = min(overheads)
    max_ov = max(overheads)

    def scale_bubble(time_overhead):
        if max_ov == min_ov:
            return 500
        return 250 + ((time_overhead - min_ov) / (max_ov - min_ov)) * 1200

    # ─────────────────────────────────────────────
    # 🔗 === Thin dashed trend line: Baseline → L6 → L5-L6 → L4-L6 → L3-L6
    # ─────────────────────────────────────────────
    # ax.plot(
    #     [throughput_drop for _, _, _, throughput_drop, _ in tradeoff_data],
    #     [bleu_gain_pct for _, bleu_gain_pct, _, _, _ in tradeoff_data],
    #     linestyle="-",
    #     linewidth=1.0,
    #     color="#6E6E6E",
    #     alpha=0.65,
    #     zorder=1
    # )

    trend_line, = ax.plot(
        [throughput_drop for _, _, _, throughput_drop, _ in tradeoff_data],
        [bleu_gain_pct for _, bleu_gain_pct, _, _, _ in tradeoff_data],
        linestyle=(0, (1, 4)),   # round dotted line
        linewidth=1.6,
        color="#5A5A5A",
        alpha=0.85,
        zorder=1
    )

    trend_line.set_dash_capstyle("round")

    # ─────────────────────────────────────────────
    # ⚙️ === Draw bubbles
    # ─────────────────────────────────────────────
    for model, bleu_gain_pct, time_overhead, throughput_drop, layers in tradeoff_data:

        color_key = MODEL2COLOR[model]
        bubble_size = 160 if model == "Baseline" else scale_bubble(time_overhead)

        ax.scatter(
            [throughput_drop], [bleu_gain_pct],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM (L4-L6)" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === Annotate inference-time overhead vs baseline (%)
    # ─────────────────────────────────────────────
    for model, bleu_gain_pct, time_overhead, throughput_drop, layers in tradeoff_data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            throughput_drop + dx,
            bleu_gain_pct + dy,
            rf"\textbf{{{time_overhead:.2f}\%}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === Labels / axis
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Throughput Drop vs Baseline (\%)}")
    ax.set_ylabel(r"\textbf{BLEU Gain vs Baseline (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1.5, 15.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

    ax.set_ylim(-0.15, 2.65)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # ─────────────────────────────────────────────
    # 🔍 === Legend
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in tradeoff_data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === Save
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_ThroughputDropX_BLEUGainPctY_TimeOverheadBubble_DashedLine_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_ThroughputDropX_BLEUGainPctY_TimeOverheadBubble_DashedLine_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOT 🔑========================🔗
# ===============================================================
plot_tdm_tradeoff_oneplot_with_line()
# ────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 9.5 TDM-FORMER TRADE-OFF BUBBLE PLOT 2 | XXX ----🎀📣 Y-axis: BLEU GAIN (%) ---- ####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9.5 TDM-FORMER TRADE-OFF BUBBLE PLOT 2 | XXX ----🎀📣 Y-axis: BLEU GAIN (%) ---- ####################
########################################################################################################################

"""
X-axis = Inference-time vs Baseline (%)
Y-axis = BLEU Gain vs Baseline (%)
Bubble size = Throughput Drop vs Baseline (%)
Text label = Throughput Drop vs Baseline (%)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================
# 1️⃣ 📊 Data: (Model, BLEU Best, BLEU Gain, BLEU Gain %, Time Overhead %, Throughput Drop %, Enhanced Layers)
# ==================================================================================================
data = [

    # 🧠 TDM-Former Trade-Off Data
    ("Baseline",        30.81, 0.00, 0.00,  0.00,  0.00, "--"),
    ("TDM L6",          31.16, 0.35, 1.14,  6.05,  4.20, "L6"),
    ("TDM L5-L6",       31.37, 0.56, 1.82,  8.37,  6.42, "L5-L6"),
    ("TDM L4-L6",       31.51, 0.70, 2.27, 12.70, 10.45, "L4-L6"),
    ("TDM L3-L6",       31.54, 0.73, 2.37, 17.84, 13.81, "L3-L6"),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — TDM-Former Trade-Off ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline_model":        "#2E2E2E",
    "TDM_L6_model":          "#EF476F",
    "TDM_L5_L6_model":       "#06D6A0",
    "TDM_L4_L6_model":       "#E49B0F",
    "TDM_L3_L6_model":       "#8338EC",
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ===============================================================
# 5️⃣🔗================ GENERATE PLOT 2 🔑====================🔗
# ===============================================================
def plot_tdm_tradeoff_oneplot_2(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    MODEL2COLOR = {
        "Baseline":    "Baseline_model",
        "TDM L6":      "TDM_L6_model",
        "TDM L5-L6":   "TDM_L5_L6_model",
        "TDM L4-L6":   "TDM_L4_L6_model",
        "TDM L3-L6":   "TDM_L3_L6_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === Plot 2 data
    # X-axis = Inference-time overhead (%)
    # Y-axis = BLEU Gain vs Baseline (%)
    # Bubble size = Throughput Drop vs Baseline (%)
    # Text label = Throughput Drop vs Baseline (%)
    # ─────────────────────────────────────────────
    tradeoff_data = []

    for model, bleu_best, bleu_gain, bleu_gain_pct, time_overhead, throughput_drop, layers in data:
        tradeoff_data.append((model, bleu_gain_pct, time_overhead, throughput_drop, layers))

    ANNOT_OFFSET = {
        ("ALL", "Baseline"):   (-0.8,  0.115),
        ("ALL", "TDM L6"):     (-1.0,  -0.22),
        ("ALL", "TDM L5-L6"):  (-1.0, -0.245),
        ("ALL", "TDM L4-L6"):  (-1.1,  -0.27),
        ("ALL", "TDM L3-L6"):  (-1.2, -0.305),
    }

    TILTED_ANNOT = {
        # ("ALL", "TDM L3-L6"): 15,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === Bubble size = throughput drop vs baseline (%)
    # ─────────────────────────────────────────────
    throughput_drops = [throughput_drop for _, _, _, throughput_drop, _ in tradeoff_data]
    min_drop = min(throughput_drops)
    max_drop = max(throughput_drops)

    def scale_bubble(throughput_drop):
        if max_drop == min_drop:
            return 500
        return 250 + ((throughput_drop - min_drop) / (max_drop - min_drop)) * 1200

    # ─────────────────────────────────────────────
    # ⚙️ === Draw bubbles
    # ─────────────────────────────────────────────
    for model, bleu_gain_pct, time_overhead, throughput_drop, layers in tradeoff_data:

        color_key = MODEL2COLOR[model]
        bubble_size = 160 if model == "Baseline" else scale_bubble(throughput_drop)

        ax.scatter(
            [time_overhead], [bleu_gain_pct],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "TDM L4-L6" else 5
        )

    # ─────────────────────────────────────────────
    # ✍️ === Annotate throughput drop vs baseline (%)
    # ─────────────────────────────────────────────
    for model, bleu_gain_pct, time_overhead, throughput_drop, layers in tradeoff_data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.03, 0.03))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)

        ax.text(
            time_overhead + dx,
            bleu_gain_pct + dy,
            rf"\textbf{{{throughput_drop:.2f}\%}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === Labels / axis
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Inference-Time Overhead vs Baseline (\%)}")
    ax.set_ylabel(r"\textbf{BLEU Gain vs Baseline (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1.5, 19.5)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18])

    ax.set_ylim(-0.15, 2.65)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # ─────────────────────────────────────────────
    # 🔍 === Legend
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _, _ in tradeoff_data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === Save
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_TimeOverheadX_BLEUGainPctY_ThroughputDropBubble_{exp_args.dataset_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"TDM_Tradeoff_TimeOverheadX_BLEUGainPctY_ThroughputDropBubble_{exp_args.dataset_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOT 2 🔑======================🔗
# ===============================================================
plot_tdm_tradeoff_oneplot_2()
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# %%
