



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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Transformers\Github\2-Results\Ablation"
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
exp_parser = argparse.ArgumentParser("IWSLT Ablation Config")

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





# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣ === READ MODEL SUMMARY ===

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




########################################################################################################################
####-------| NOTE 7. DEFINE FILE PATH | XXX --------------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== 7️⃣.1️⃣ Define Train/Test log file paths ==========================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def build_log_path(curve_id, folder_name):
    # plot_name    = getattr(exp_args, f"Plot{curve_id}")
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
        "Results",                 # 📌📌 Result Folder name 🔖🔖
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
# 📌📌 ======== 7️⃣.2️⃣ Define model_summary_full.txt file paths ==================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def build_model_summary_path(curve_id, folder_name):
    mode_name    = getattr(exp_args, f"mode_name_Plot{curve_id}")
    net_name     = getattr(exp_args, f"net_Plot{curve_id}")
    dataset_name = getattr(exp_args, f"dataset_name_Plot{curve_id}")

    file_name = f"{net_name}_{dataset_name}_{exp_args.optimizer1}_{mode_name}_model_summary_full.txt"

    return os.path.join(
        DATA_TEST_PATH,
        folder_name,
        "Results_Architecture_Summary",       # 📌📌 Result Folder name 🔖🔖
        dataset_name,
        net_name,
        file_name
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define all model summary paths ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────

Summary_Plot_1 = build_model_summary_path("1", "1.1-Transformer_Baselines_DE_EN-Baseline")  # Baseline
Summary_Plot_2 = build_model_summary_path("2", "1.2-Transformer_Baselines_DE_EN-L6")        # L6
Summary_Plot_3 = build_model_summary_path("3", "1.3-Transformer_Baselines_DE_EN-L5_L6")     # L5-L6
Summary_Plot_4 = build_model_summary_path("4", "1.4-Transformer_Baselines_DE_EN-L4_L6")     # L4-L6
Summary_Plot_5 = build_model_summary_path("5", "1.5-Transformer_Baselines_DE_EN-L3_L6")     # L3-L6


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print model summary file paths sanity check ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────

print(f"\n📄 MODEL SUMMARY file paths:")
print("─" * 100)

all_summary_paths = {
    "DE→EN | Baseline":       Summary_Plot_1,
    "DE→EN | TDM L6":         Summary_Plot_2,
    "DE→EN | TDM L5-L6":      Summary_Plot_3,
    "DE→EN | TDM L4-L6":      Summary_Plot_4,
    "DE→EN | TDM L3-L6":      Summary_Plot_5,
}

for name, path in all_summary_paths.items():
    print(f"📄 {name}:")
    print(f"   {path}\n")

print("─" * 100)
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 8. READ LOGS| XXX -----🔑🔗------------------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 8. READ LOGS| XXX -----🔑🔗------------------------------------------------------####################
########################################################################################################################



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== 8️⃣.1️⃣ TEST AND TRAIN DATA =======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== 8️⃣.1️⃣ TEST AND TRAIN DATA =======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ 🧩🧠 === Test ♻️ ♻️ ======================================================================= 
# ────────────────────────────────────────────────────────────────────────────────────────────────
if exp_args.evaluation_mode == "test": 

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ === Read TEST metrics: Test Loss + BLEU + Best BLEU + Best Epoch
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    BASE_epochs_test, BASE_test_loss, BASE_bleu, BASE_best_bleu, BASE_best_epoch = read_test_metrics(Path_Plot_1)

    L6_epochs_test, L6_test_loss, L6_bleu, L6_best_bleu, L6_best_epoch = read_test_metrics(Path_Plot_2)

    L5_L6_epochs_test, L5_L6_test_loss, L5_L6_bleu, L5_L6_best_bleu, L5_L6_best_epoch = read_test_metrics(Path_Plot_3)

    L4_L6_epochs_test, L4_L6_test_loss, L4_L6_bleu, L4_L6_best_bleu, L4_L6_best_epoch = read_test_metrics(Path_Plot_4)

    L3_L6_epochs_test, L3_L6_test_loss, L3_L6_bleu, L3_L6_best_bleu, L3_L6_best_epoch = read_test_metrics(Path_Plot_5)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🏆 Print TEST summary with BLEU gain
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    print("\n🏆 TEST Best BLEU Summary: DE→EN TDM Depth Ablation")
    print("─" * 115)
    print(f"{'Variant':<18} | {'BLEU':>8} | {'Gain':>8} | {'Rel. Gain':>10} | {'Best Epoch':>10}")
    print("─" * 115)

    summary_rows = [
        ("Baseline", BASE_best_bleu, BASE_best_epoch),
        ("TDM L6", L6_best_bleu, L6_best_epoch),
        ("TDM L5-L6", L5_L6_best_bleu, L5_L6_best_epoch),
        ("TDM L4-L6", L4_L6_best_bleu, L4_L6_best_epoch),
        ("TDM L3-L6", L3_L6_best_bleu, L3_L6_best_epoch),
    ]

    for variant, bleu, epoch in summary_rows:

        if variant == "Baseline":
            abs_gain = 0.00
            rel_gain = 0.00
        else:
            abs_gain, rel_gain = compute_bleu_gains(bleu, BASE_best_bleu)

        print(
            f"{variant:<18} | "
            f"{bleu:>8.2f} | "
            f"{abs_gain:>+8.2f} | "
            f"{rel_gain:>+9.2f}% | "
            f"{epoch:>10}"
        )

    print("─" * 115)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 💾 Save TEST summary to TXT file
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    summary_txt_path = os.path.join(DATA_TEST_PATH, "TDM_Former_DE_EN_Depth_Ablation_TEST_BLEU_Gains.txt")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("🏆 TEST Best BLEU Summary: DE→EN TDM Depth Ablation\n")
        f.write("─" * 115 + "\n")
        f.write(f"{'Variant':<18} | {'BLEU':>8} | {'Gain':>8} | {'Rel. Gain':>10} | {'Best Epoch':>10}\n")
        f.write("─" * 115 + "\n")

        for variant, bleu, epoch in summary_rows:

            if variant == "Baseline":
                abs_gain = 0.00
                rel_gain = 0.00
            else:
                abs_gain, rel_gain = compute_bleu_gains(bleu, BASE_best_bleu)

            f.write(
                f"{variant:<18} | "
                f"{bleu:>8.2f} | "
                f"{abs_gain:>+8.2f} | "
                f"{rel_gain:>+9.2f}% | "
                f"{epoch:>10}\n"
            )

        f.write("─" * 115 + "\n")

    print(f"✅ Saved TEST BLEU summary to: {summary_txt_path}")


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ 🧩🧠 === Train ♻️ ♻️ ======================================================================= 
# ────────────────────────────────────────────────────────────────────────────────────────────────
elif exp_args.evaluation_mode == "train": 

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ === Read TRAIN metrics: Train Loss + LR + Best/Lowest Train Loss + Best Epoch
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    BASE_epochs_train, BASE_train_loss, BASE_lr, BASE_best_loss, BASE_best_epoch = read_train_metrics(Path_Plot_1)

    L6_epochs_train, L6_train_loss, L6_lr, L6_best_loss, L6_best_epoch = read_train_metrics(Path_Plot_2)

    L5_L6_epochs_train, L5_L6_train_loss, L5_L6_lr, L5_L6_best_loss, L5_L6_best_epoch = read_train_metrics(Path_Plot_3)

    L4_L6_epochs_train, L4_L6_train_loss, L4_L6_lr, L4_L6_best_loss, L4_L6_best_epoch = read_train_metrics(Path_Plot_4)

    L3_L6_epochs_train, L3_L6_train_loss, L3_L6_lr, L3_L6_best_loss, L3_L6_best_epoch = read_train_metrics(Path_Plot_5)


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🏆 Print TRAIN summary
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    print("\n🏆 TRAIN Best/Lowest Loss Summary: DE→EN TDM Depth Ablation")
    print("─" * 90)

    print(f"Baseline   : Loss={BASE_best_loss:.3f} | Epoch={BASE_best_epoch}")
    print(f"TDM L6     : Loss={L6_best_loss:.3f} | Epoch={L6_best_epoch}")
    print(f"TDM L5-L6  : Loss={L5_L6_best_loss:.3f} | Epoch={L5_L6_best_epoch}")
    print(f"TDM L4-L6  : Loss={L4_L6_best_loss:.3f} | Epoch={L4_L6_best_epoch}")
    print(f"TDM L3-L6  : Loss={L3_L6_best_loss:.3f} | Epoch={L3_L6_best_epoch}")

    print("─" * 90)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(f"Unknown Evaluation mode: {exp_args.evaluation_mode}")
# =====================================================================================================




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== 8️⃣.2️⃣ MODEL SUMMARY DATA ========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ======== 8️⃣.2️⃣ MODEL SUMMARY DATA ========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ 🧩🧠 === Read model summary files ♻️ ♻️ =====================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────

BASE_summary  = read_model_summary(Summary_Plot_1)

L6_summary    = read_model_summary(Summary_Plot_2)

L5_L6_summary = read_model_summary(Summary_Plot_3)

L4_L6_summary = read_model_summary(Summary_Plot_4)

L3_L6_summary = read_model_summary(Summary_Plot_5)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ 🧩🧠 === Collect model summaries ============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────

model_summary_rows = [
    ("Baseline",  BASE_summary),
    ("TDM L6",    L6_summary),
    ("TDM L5-L6", L5_L6_summary),
    ("TDM L4-L6", L4_L6_summary),
    ("TDM L3-L6", L3_L6_summary),
]


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣ 🏆 === Print compact model summary ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────

print("\n📄 MODEL SUMMARY: DE→EN TDM Depth Ablation")
print("─" * 180)
print(
    f"{'Variant':<12} | "
    f"{'Dataset':<14} | "
    f"{'Dir.':<8} | "
    # f"{'Params(M)':>9} | "
    f"{'Total Params':>12} | "
    f"{'MACs(G)':>8} | "
    f"{'FLOPs(G)':>9} | "
    f"{'Act.':<6} | "
    f"{'Opt.':<6} | "
    f"{'LR':>8} | "
    f"{'Smooth':>7} | "
    f"{'Layers':<8}"
)
print("─" * 180)

for variant, summary in model_summary_rows:

    if summary is None:
        print(f"{variant:<12} | ⚠️ Missing summary file")
        continue

    dataset_name = summary["dataset_info"]["dataset_name"]
    direction    = summary["dataset_info"]["translation_direction"]

    params_M = summary["complexity"]["params_M"]
    macs_G   = summary["complexity"]["macs_G"]
    flops_G  = summary["complexity"]["flops_G"]
    total_params = summary["total_params"]

    activation = summary["optimizer_activation"]["activation"]
    optimizer  = summary["optimizer_activation"]["optimizer"]
    lr         = summary["optimizer_activation"]["lr"]
    smoothing  = summary["optimizer_activation"]["smoothing"]

    if summary["tdm_params"] is None:
        layers = "Base"
    else:
        layers = f"L{7 - summary['tdm_params']['tdm_num_layers']}-L6" if summary["tdm_params"]["tdm_num_layers"] > 1 else "L6"

    print(
        f"{variant:<12} | "
        f"{dataset_name:<14} | "
        f"{direction:<8} | "
        # f"{params_M:>9.4f} | "
        f"{total_params:>12,} | "
        f"{macs_G:>8.4f} | "
        f"{flops_G:>9.4f} | "
        f"{activation:<6} | "
        f"{optimizer:<6} | "
        f"{lr:>8.5f} | "
        f"{smoothing:>7.2f} | "
        f"{layers:<8}"
    )

print("─" * 180)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣ 🏆 === Print TDM parameter summary ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────

print("\n🔧 TDM PARAMETER SUMMARY: DE→EN Depth Ablation")
print("─" * 140)
print(
    f"{'Variant':<12} | "
    f"{'alpha':>6} | "
    f"{'amp':>6} | "
    f"{'num_layers':>10} | "
    f"{'transition':>10} | "
    f"{'rotation':>8} | "
    f"{'insert':<8} | "
    f"{'scale_init':>10} | "
    f"{'start':>6} | "
    f"{'full':>6} | "
    f"{'gate_type':<22}"
)
print("─" * 140)

for variant, summary in model_summary_rows:

    if summary is None:
        print(f"{variant:<12} | ⚠️ Missing summary file")
        continue

    if summary["tdm_params"] is None:
        print(f"{variant:<12} | Baseline Transformer: no TDM parameters")
        continue

    t = summary["tdm_params"]

    print(
        f"{variant:<12} | "
        f"{t['tdm_alpha']:>6.2f} | "
        f"{t['tdm_transition_amplification']:>6.2f} | "
        f"{t['tdm_num_layers']:>10} | "
        f"{str(t['tdm_transition']):>10} | "
        f"{str(t['tdm_rotation']):>8} | "
        f"{t['tdm_insertion_type']:<8} | "
        f"{t['tdm_scale_init']:>10.2f} | "
        f"{t['tdm_start_epoch']:>6} | "
        f"{t['tdm_full_epoch']:>6} | "
        f"{t['tdm_gate_type']:<22}"
    )

print("─" * 140)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣ 💾 === Save model summary to TXT file =======================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────

model_summary_txt_path = os.path.join(DATA_TEST_PATH, "TDM_Former_DE_EN_Depth_Ablation_MODEL_Summary.txt")

with open(model_summary_txt_path, "w", encoding="utf-8") as f:

    f.write("📄 MODEL SUMMARY: DE→EN TDM Depth Ablation\n")
    f.write("─" * 180 + "\n")
    f.write(
        f"{'Variant':<12} | "
        f"{'Dataset':<14} | "
        f"{'Dir.':<8} | "
        f"{'Train':>8} | "
        f"{'Test':>8} | "
        # f"{'Params(M)':>9} | "
        f"{'Total Params':>12} | "
        f"{'MACs(G)':>8} | "
        f"{'FLOPs(G)':>9} | "
        f"{'Act.':<6} | "
        f"{'Opt.':<6} | "
        f"{'LR':>8} | "
        f"{'Smooth':>7} | "
        f"{'Enc':>3} | "
        f"{'Dec':>3} | "
        f"{'Emb':>4} | "
        f"{'MaxPos':>6} | "
        f"{'Drop':>5}\n"
    )
    f.write("─" * 180 + "\n")

    for variant, summary in model_summary_rows:

        if summary is None:
            f.write(f"{variant:<12} | Missing summary file\n")
            continue

        d  = summary["dataset_info"]
        c  = summary["complexity"]
        tp = summary["total_params"]
        oa = summary["optimizer_activation"]
        fs = summary["fairseq_params"]

        f.write(
            f"{variant:<12} | "
            f"{d['dataset_name']:<14} | "
            f"{d['translation_direction']:<8} | "
            f"{d['train_samples']:>8} | "
            f"{d['test_samples']:>8} | "
            # f"{c['params_M']:>9.4f} | "
            f"{tp:>12,} | "
            f"{c['macs_G']:>8.4f} | "
            f"{c['flops_G']:>9.4f} | "
            f"{oa['activation']:<6} | "
            f"{oa['optimizer']:<6} | "
            f"{oa['lr']:>8.5f} | "
            f"{oa['smoothing']:>7.2f} | "
            f"{fs['encoder_layers']:>3} | "
            f"{fs['decoder_layers']:>3} | "
            f"{fs['encoder_embed_dim']:>4} | "
            f"{fs['max_target_positions']:>6} | "
            f"{fs['dropout']:>5.2f}\n"
        )

    f.write("─" * 180 + "\n\n")

    f.write("🔧 TDM PARAMETER SUMMARY\n")
    f.write("─" * 140 + "\n")
    f.write(
        f"{'Variant':<12} | "
        f"{'alpha':>6} | "
        f"{'amp':>6} | "
        f"{'num_layers':>10} | "
        f"{'transition':>10} | "
        f"{'rotation':>8} | "
        f"{'insert':<8} | "
        f"{'scale_init':>10} | "
        f"{'start':>6} | "
        f"{'full':>6} | "
        f"{'gate_type':<22}\n"
    )
    f.write("─" * 140 + "\n")

    for variant, summary in model_summary_rows:

        if summary is None:
            f.write(f"{variant:<12} | Missing summary file\n")
            continue

        if summary["tdm_params"] is None:
            f.write(f"{variant:<12} | Baseline Transformer: no TDM parameters\n")
            continue

        t = summary["tdm_params"]

        f.write(
            f"{variant:<12} | "
            f"{t['tdm_alpha']:>6.2f} | "
            f"{t['tdm_transition_amplification']:>6.2f} | "
            f"{t['tdm_num_layers']:>10} | "
            f"{str(t['tdm_transition']):>10} | "
            f"{str(t['tdm_rotation']):>8} | "
            f"{t['tdm_insertion_type']:<8} | "
            f"{t['tdm_scale_init']:>10.2f} | "
            f"{t['tdm_start_epoch']:>6} | "
            f"{t['tdm_full_epoch']:>6} | "
            f"{t['tdm_gate_type']:<22}\n"
        )

    f.write("─" * 140 + "\n")

print(f"✅ Saved MODEL summary to: {model_summary_txt_path}")

# =====================================================================================================








# %% 

########################################################################################################################
####-------| NOTE 9. PLOT FUNCTION | XXX -----🔑📐🔗-----------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9. PLOT FUNCTION | XXX -----🔑📐🔗-----------------------------------------------####################
########################################################################################################################




# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====9️⃣.1️⃣ BLEU ====🔗
# ===============================================================
# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====9️⃣.1️⃣ BLEU ====🔗
# ===============================================================



from matplotlib.lines import Line2D

def plot_tdm_depth_ablation_bleu_vs_epoch(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅰️🔼 🎨 === Colors / styles: DE→EN TDM depth ablation
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    COLORS = {
        "Baseline": "#2E2E2E",
        "L6":       "#06D6A0",
        "L5_L6":    "#3A86FF",
        "L4_L6":    "#E49B0F",
        "L3_L6":    "#8338EC",
    }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅱️🔼🔧 === One figure: BLEU vs Epoch for DE→EN depth ablation
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 1️⃣📈 ========= Plot BLEU curves ===============================
    # ────────────────────────────────────────────────────────────────
    ax.plot(BASE_epochs_test, BASE_bleu,
            color=COLORS["Baseline"], linewidth=2.0, linestyle="--",
            alpha=1.0, label=r"\textbf{Baseline}", zorder=4)

    ax.plot(L6_epochs_test, L6_bleu,
            color=COLORS["L6"], linewidth=2.0, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L6)}", zorder=5)

    ax.plot(L5_L6_epochs_test, L5_L6_bleu,
            color=COLORS["L5_L6"], linewidth=2.0, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L5--L6)}", zorder=6)

    ax.plot(L4_L6_epochs_test, L4_L6_bleu,
            color=COLORS["L4_L6"], linewidth=2.3, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L4--L6)}", zorder=8)

    ax.plot(L3_L6_epochs_test, L3_L6_bleu,
            color=COLORS["L3_L6"], linewidth=2.0, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L3--L6)}", zorder=7)

    # ────────────────────────────────────────────────────────────────
    # 2️⃣📍 ========= Mark best BLEU points ==========================
    # ────────────────────────────────────────────────────────────────
    ax.scatter([BASE_best_epoch], [BASE_best_bleu],
               s=30, facecolors='white', edgecolors=COLORS["Baseline"],
               linewidths=1.6, zorder=10)

    ax.scatter([L6_best_epoch], [L6_best_bleu],
               s=30, facecolors='white', edgecolors=COLORS["L6"],
               linewidths=1.6, zorder=10)

    ax.scatter([L5_L6_best_epoch], [L5_L6_best_bleu],
               s=30, facecolors='white', edgecolors=COLORS["L5_L6"],
               linewidths=1.6, zorder=10)

    ax.scatter([L4_L6_best_epoch], [L4_L6_best_bleu],
               s=30, facecolors='white', edgecolors=COLORS["L4_L6"],
               linewidths=1.6, zorder=20)

    ax.scatter([L3_L6_best_epoch], [L3_L6_best_bleu],
               s=30, facecolors='white', edgecolors=COLORS["L3_L6"],
               linewidths=1.6, zorder=10)

    # ────────────────────────────────────────────────────────────────
    # 3️⃣⚙️ ========= Axis labels / control ==========================
    # ────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Epoch}")
    ax.set_ylabel(r"\textbf{BLEU}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1, 51)
    # ax.set_xticks(range(0, 50, 10))
    ax.set_xticks([0, 10, 20, 30, 40, 50])

    ax.set_ylim(27.3, 31.7)
    ax.set_yticks([28, 29, 30, 31])

    # ────────────────────────────────────────────────────────────────
    # 4️⃣🔍 ========= Legend =========================================
    # ────────────────────────────────────────────────────────────────
    leg = ax.legend(
        fontsize=exp_args.legend_font,
        frameon=False,
        ncol=1,        
        loc="upper left",
        # bbox_to_anchor=(0.00, 0.94),
        bbox_to_anchor=(0.00, 1.0),
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.35,
        borderaxespad=0.2,
    )

    leg._legend_box.align = "left"
    leg.set_zorder(100)

    # ────────────────────────────────────────────────────────────────
    # 5️⃣📦 ======== Save figures ====================================
    # ────────────────────────────────────────────────────────────────
    file_name = "TDM_Former_DE_EN_Depth_Ablation_BLEU_vs_Epoch"

    fig.savefig(
        os.path.join(save_dir, f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )

    fig.savefig(
        os.path.join(save_dir, f"{file_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
plot_tdm_depth_ablation_bleu_vs_epoch()
# ────────────────────────────────────────────────────────────────






# %% 




# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====9️⃣.1️⃣ LOSS ====🔗
# ===============================================================
# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====9️⃣.1️⃣ LOSS ====🔗
# ===============================================================


from matplotlib.lines import Line2D

def plot_tdm_depth_ablation_loss_vs_epoch(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅰️🔼 🎨 === Colors / styles: DE→EN TDM depth ablation
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    COLORS = {
        "Baseline": "#2E2E2E",
        # "L6":       "#06D6A0",
        "L6":       "#EF476F",
        # "L5_L6":    "#3A86FF",
        "L5_L6":    "#06D6A0",
        "L4_L6":    "#E49B0F",
        "L3_L6":    "#8338EC",
    }

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅰️.1️⃣🔧 === Helper: get lowest loss and its epoch
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def get_best_loss_point(epochs, losses):
        best_idx = losses.index(min(losses))
        return epochs[best_idx], losses[best_idx]

    BASE_best_loss_epoch, BASE_best_loss = get_best_loss_point(BASE_epochs_test, BASE_test_loss)
    L6_best_loss_epoch, L6_best_loss = get_best_loss_point(L6_epochs_test, L6_test_loss)
    L5_L6_best_loss_epoch, L5_L6_best_loss = get_best_loss_point(L5_L6_epochs_test, L5_L6_test_loss)
    L4_L6_best_loss_epoch, L4_L6_best_loss = get_best_loss_point(L4_L6_epochs_test, L4_L6_test_loss)
    L3_L6_best_loss_epoch, L3_L6_best_loss = get_best_loss_point(L3_L6_epochs_test, L3_L6_test_loss)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 🅱️🔼🔧 === One figure: Test Loss vs Epoch for DE→EN depth ablation
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 1️⃣📉 ========= Plot LOSS curves ===============================
    # ────────────────────────────────────────────────────────────────
    ax.plot(BASE_epochs_test, BASE_test_loss,
            color=COLORS["Baseline"], linewidth=2.0, linestyle="--",
            alpha=1.0, label=r"\textbf{Baseline}", zorder=4)

    ax.plot(L6_epochs_test, L6_test_loss,
            color=COLORS["L6"], linewidth=2.0, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L6)}", zorder=5)

    ax.plot(L5_L6_epochs_test, L5_L6_test_loss,
            color=COLORS["L5_L6"], linewidth=2.0, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L5--L6)}", zorder=6)

    ax.plot(L4_L6_epochs_test, L4_L6_test_loss,
            color=COLORS["L4_L6"], linewidth=2.3, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L4--L6)}", zorder=8)

    ax.plot(L3_L6_epochs_test, L3_L6_test_loss,
            color=COLORS["L3_L6"], linewidth=2.0, linestyle="-",
            alpha=1.0, label=r"\textbf{TDM (L3--L6)}", zorder=7)

    # ────────────────────────────────────────────────────────────────
    # 2️⃣📍 ========= Mark lowest LOSS points ========================
    # ────────────────────────────────────────────────────────────────
    ax.scatter([BASE_best_loss_epoch], [BASE_best_loss],
               s=30, facecolors='white', edgecolors=COLORS["Baseline"],
               linewidths=1.6, zorder=10)

    ax.scatter([L6_best_loss_epoch], [L6_best_loss],
               s=30, facecolors='white', edgecolors=COLORS["L6"],
               linewidths=1.6, zorder=10)

    ax.scatter([L5_L6_best_loss_epoch], [L5_L6_best_loss],
               s=30, facecolors='white', edgecolors=COLORS["L5_L6"],
               linewidths=1.6, zorder=10)

    ax.scatter([L4_L6_best_loss_epoch], [L4_L6_best_loss],
               s=30, facecolors='white', edgecolors=COLORS["L4_L6"],
               linewidths=1.6, zorder=20)

    ax.scatter([L3_L6_best_loss_epoch], [L3_L6_best_loss],
               s=30, facecolors='white', edgecolors=COLORS["L3_L6"],
               linewidths=1.6, zorder=10)

    # ────────────────────────────────────────────────────────────────
    # 3️⃣⚙️ ========= Axis labels / control ==========================
    # ────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Epoch}")
    ax.set_ylabel(r"\textbf{Test Loss}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-1, 51)
    ax.set_xticks([0, 10, 20, 30, 40, 50])

    # Leave y-axis automatic for loss because loss range can differ
    ax.set_ylim(2.93, 3.37)
    ax.set_yticks([3.0, 3.1, 3.2, 3.3])

    # ────────────────────────────────────────────────────────────────
    # 4️⃣🔍 ========= Legend =========================================
    # ────────────────────────────────────────────────────────────────
    leg = ax.legend(
        fontsize=exp_args.legend_font,
        frameon=False,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.35,
        borderaxespad=0.2,
    )

    leg._legend_box.align = "left"
    leg.set_zorder(100)

    # ────────────────────────────────────────────────────────────────
    # 5️⃣📦 ======== Save figures ====================================
    # ────────────────────────────────────────────────────────────────
    file_name = "TDM_Former_DE_EN_Depth_Ablation_LOSS_vs_Epoch"

    fig.savefig(
        os.path.join(save_dir, f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )

    fig.savefig(
        os.path.join(save_dir, f"{file_name}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
plot_tdm_depth_ablation_loss_vs_epoch()
# ────────────────────────────────────────────────────────────────
# %%
