





# %%  


#####-------------------------------- NOTE MAIN IWSLT17 ZH-EN NOTE --------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
###################################🔗 MAIN | TRAIN | TEST LOOP 🔗########################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE MAIN IWSLT17 ZH-EN NOTE --------------------------------------------------#####



# 📄  main_IWSLT17Zh_En.py
########################################################################################################################
####-------| NOTE 1.1 IMPORTS LIBRARIES  | XXX -------------------------------------------------------##################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 === Enable flexible CUDA memory allocation to reduce fragmentation ===
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ======================================================================================================
# 📜 === Core Libraries ===
# ======================================================================================================
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import random
import math
import re
import glob
from copy import deepcopy
import sacrebleu  # ✅ for BLEU evaluation
from timm.loss import LabelSmoothingCrossEntropy
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ======================================================================================================
# 📜 === Fairseq & Supporting Imports ===
# ======================================================================================================
import importlib.metadata as importlib_metadata
import fairseq  # type: ignore
from fairseq.data import Dictionary, data_utils, iterators          # type: ignore
from fairseq.tasks.translation import TranslationTask               # type: ignore
from fairseq import options                                         # type: ignore
from fairseq.dataclass.utils import convert_namespace_to_omegaconf  # type: ignore                                             
from fairseq import utils                                           # type: ignore 
import hydra                                                        # type: ignore  
# ────────────────────────────────────────────────────────────────────────────────────────────────
from ptflops import get_model_complexity_info
from calflops import calculate_flops
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ======================================================================================================
# ♻️ === Print environment summary for sanity check ===
# ======================================================================================================
print("Python:", sys.version)
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Fairseq:", fairseq.__version__)
print("OmegaConf:", importlib_metadata.version("omegaconf"))
print("Hydra-Core:", importlib_metadata.version("hydra-core"))
# ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 1.2. DEFINE PROJECT PATH | XXX ---------------------------------------------------####################
########################################################################################################################

# ✅ Define working directory
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Transformers\Transformer_Baselines_ZH_EN"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ✅ Define absolute paths
PROJECT_PATH = Project_PATH
MODELS_PATH = os.path.join(Project_PATH, "models")
ACTIVATION_PATH = os.path.join(Project_PATH, "activation")


# ✅ Ensure necessary paths are in sys.path
for path in [PROJECT_PATH, MODELS_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# ✅ Print updated sys.path for debugging
print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)
# ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 1.3. OTHER IMPORTS | XXX ---------------------------------------------------------####################
########################################################################################################################

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput parser safe for afno because of --fno-bias, etc  =======================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣✅ Import parser from parser_IWSLT17Zh_En.py
from parser_IWSLT17Zh_En import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
# args, unknown = parser.parse_known_args()

# ✅ IMPORTANT: Do NOT read Jupyter / VSCode kernel arguments
# This prevents the "--f" ambiguity issue
exp_args = parser.parse_args(args=[])

num_aug_splits = exp_args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")

# ────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣✅ Import Fairseq parser/config
from fairseq_config_IWSLT17Zh_En import get_fairseq_parser
# ✅ Build Fairseq configuration
fairseq_args, cfg = get_fairseq_parser(exp_args)


num_aug_splits = exp_args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")

print(f"✅ Encoder Embed Dim: {fairseq_args.encoder_embed_dim}")
print(f"✅ Decoder Layers: {fairseq_args.decoder_layers}")
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣✅ Import model summary utility
from models.model_summary.model_summary import save_model_summary
from models.model_summary.model_summary_full import save_model_summary_full

print("✅ model_summary.py loaded successfully")
print("✅ model_summary_full.py loaded successfully")
# ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 2. SEEDING FOR REPRODUCIBILITY | XXX ---------------------------------------------####################
########################################################################################################################

# ────────────────────────────────────────────────────────────────────────────────────────────────
def set_seed_torch(seed):
    torch.manual_seed(seed)
# ────────────────────────────────────────────────────────────────────────────────────────────────
def set_seed_main(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True      # ✅ Default: "False" for Faster, non-deterministic kernels | "True" to Ensure deterministic behavior for CuDNN (Slower)
    torch.backends.cudnn.benchmark = False         # ✅ Default: "True" for Autotune kernels for performance   | "False" Disable CuDNN's autotuning for reproducibility (Slower)

    torch.backends.cuda.matmul.allow_tf32 = False  # ✅ Disable TF32 (strict reproducibility)
    torch.backends.cudnn.allow_tf32 = False        # ✅ Disable TF32 (strict reproducibility)

    # torch.use_deterministic_algorithms(True, warn_only=True)

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============= Define Seed =============
seed1, seed2 = exp_args.seed1, exp_args.seed2
set_seed_torch(seed1)  
set_seed_main(seed2)  
# ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 3.1. PROCESSING DATASET 1 | XXX ---- TOKENIZATION & DATASET PREPARATION |---------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 3.1. PROCESSING DATASET 1 | XXX ------------| CHINESE→ENGLISH |-------------------####################
########################################################################################################################

"""
Stage 1:
Train SentencePiece tokenizer, encode the raw Chinese→English corpus,
and create aligned train/test text splits for neural machine translation.
#  ======== 🔖🔖 Creates both direction 🔖🔖==========================
"""

# ─────────────────────────────────────────────────────────────────────────────────────────────────
print(
    f"🔎🔎 run_dataset_tokenization: {exp_args.run_dataset_tokenization} "
    f"{'(RUNNING)✔️✔️' if exp_args.run_dataset_tokenization else '(SKIPPED)❌❌'}"
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
if exp_args.run_dataset_tokenization:
    from datasets import load_dataset    
    import sentencepiece as spm

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Path for ZH→EN dataset
    DATA_PATH = r"C:\Users\emeka\Research\ModelCUDA\Transformers\Transformer_Baselines_ZH_EN\Dataset"
    os.makedirs(DATA_PATH, exist_ok=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  1️⃣ ======== Download / load IWSLT17 EN↔ZH from Hugging Face ====================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Download to your folder (as the Hugging Face cache)
    en_zh = load_dataset(
        "IWSLT/iwslt2017", "iwslt2017-en-zh",
        cache_dir=DATA_PATH,
        trust_remote_code=True
    )
    zh_en = load_dataset(
        "IWSLT/iwslt2017", "iwslt2017-zh-en",
        cache_dir=DATA_PATH,
        trust_remote_code=True
    )  # optional
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  2️⃣ ======== Dataset verification ===============================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ sanity check (should show ~231k train)
    print(f"EN→ZH dataset: {en_zh}")
    print(f"ZH→EN dataset: {zh_en}")

    print("EN→ZH splits:", list(en_zh.keys()), {k: len(en_zh[k]) for k in en_zh.keys()})
    print("ZH→EN splits:", list(zh_en.keys()), {k: len(zh_en[k]) for k in zh_en.keys()})

    # optional assertions
    assert set(en_zh.keys()) == {"train", "validation", "test"}
    assert set(zh_en.keys()) == {"train", "validation", "test"}
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  3️⃣ ======== Export official train/validation/test splits ======================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Export official IWSLT17 splits to plain text (EN/ZH) under raw_zh_en_dataset_path
    def dump_split(ds_dict, split, src, tgt, out_dir):
        src_fp = os.path.join(out_dir, f"{split}.{src}")
        tgt_fp = os.path.join(out_dir, f"{split}.{tgt}")
        with open(src_fp, "w", encoding="utf-8") as fs, open(tgt_fp, "w", encoding="utf-8") as ft:
            for row in ds_dict[split]:
                tr = row["translation"]
                fs.write((tr[src] or "").strip() + "\n")
                ft.write((tr[tgt] or "").strip() + "\n")
        print(f"Wrote {split}: {src_fp} | {tgt_fp}")

    for split in ["train", "validation", "test"]:
        dump_split(en_zh, split, "en", "zh", DATA_PATH)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  4️⃣ ========  Train SentencePiece model =========================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────     

    print("🧠 Training SentencePiece model for EN→ZH...")

    # Use TRAIN split for SPM training
    SRC_EN = os.path.join(DATA_PATH, "train.en")
    TGT_ZH = os.path.join(DATA_PATH, "train.zh")

    spm_model_prefix = os.path.join(DATA_PATH, "spm_enzh")

    spm.SentencePieceTrainer.Train(
        input=f"{SRC_EN},{TGT_ZH}",
        model_prefix=spm_model_prefix,
        vocab_size=10000,
        character_coverage=0.9995,   # ✅ --character_coverage for rich charcater langauage: Japanese or Chinese = 0.9995 | other languages with small character set: 1.0
        model_type="bpe"
    )
    print(f"✅ SentencePiece model saved to: {spm_model_prefix}.model / .vocab")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  5️⃣ ========  Encode files ======================================================================
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 

    print("⚙️ Encoding dataset...")

    # ✅ Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor(model_file=f"{spm_model_prefix}.model")

    def encode_file(in_file, out_file):
        with open(in_file, encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")

    # ✅ Encode all official splits (no custom re-split)
    for split in ["train", "validation", "test"]:
        encode_file(os.path.join(DATA_PATH, f"{split}.en"), os.path.join(DATA_PATH, f"{split}.spm.en"))
        encode_file(os.path.join(DATA_PATH, f"{split}.zh"), os.path.join(DATA_PATH, f"{split}.spm.zh"))
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  6️⃣ ======== Use official train/validation/test splits =========================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    print(f"✅ Using official IWSLT17 train/validation/test splits.")

    print(f"✅ Encoded EN↔ZH data saved under: {DATA_PATH}")
    print(f"✅ Next: run fairseq-preprocess to binarize (on *.spm.en/*.spm.zh).")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 3.2. PROCESSING DATASET 2 | XXX ----- FAIRSEQ PREPROCESSING & BINARIZATION |------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 3.2. PROCESSING DATASET 2 | XXX ------------| CHINESE→ENGLISH |-------------------####################
########################################################################################################################

"""
Stage 2:
Preprocess the tokenized Chinese→English dataset using fairseq-preprocess
to build vocabularies and generate Fairseq binary training files.
"""

# ─────────────────────────────────────────────────────────────────────────────────────────────────
print(
    f"🔎🔎 run_fairseq_preprocessing: {exp_args.run_fairseq_preprocessing} "
    f"{'(RUNNING)✔️✔️' if exp_args.run_fairseq_preprocessing else '(SKIPPED)❌❌'}"
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
if exp_args.run_fairseq_preprocessing:
        
    import subprocess


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  1️⃣ ======== Check dataset files before preprocessing ==========================================
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 

    # ✅ Same folder you used above
    DATA_DIR = r"C:\Users\emeka\Research\ModelCUDA\Transformers\Transformer_Baselines_ZH_EN\Dataset"
    DESTDIR  = r"data-bin\iwslt17.zh-en"  # name it how you like
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    required_files = [
        os.path.join(DATA_DIR, "train.spm.en"),
        os.path.join(DATA_DIR, "train.spm.zh"),
        os.path.join(DATA_DIR, "validation.spm.en"),
        os.path.join(DATA_DIR, "validation.spm.zh"),
        os.path.join(DATA_DIR, "test.spm.en"),
        os.path.join(DATA_DIR, "test.spm.zh"),
    ]

    for f in required_files:
        assert os.path.exists(f), f"❌ Missing file: {f}"

    print("✅ All SentencePiece files found.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  2️⃣ ======== Configure preprocessing ============================================================
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    print("🚀 Running fairseq-preprocess for ZH→EN...")

    cmd = [
        "fairseq-preprocess",
        "--source-lang", "zh",             # ✅Source = Chinese
        "--target-lang", "en",             # ✅Target = English
        # use the SPM-encoded prefixes you just created
        "--trainpref", os.path.join(DATA_DIR, "train.spm"),
        "--validpref", os.path.join(DATA_DIR, "validation.spm"),
        "--testpref",  os.path.join(DATA_DIR, "test.spm"),
        "--destdir", DESTDIR,
        "--workers", "8"
    ]
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  3️⃣ ======== Run preprocessing ==================================================================
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 

    subprocess.run(cmd, check=True)
    print(f"✅ Preprocessing complete! Binarized ZH→EN data saved in {DESTDIR}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



 


########################################################################################################################
####-------| NOTE 4. FAIRSEQ SETUP, ACTIVATION PATCHING | XXX --------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  1️⃣ ========  4.1 Device Configuration =========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣ ========  4.2 Custom Activation Registration & Patching ====================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
# Ensure activation folder is in sys.path
if ACTIVATION_PATH not in sys.path:
    sys.path.append(ACTIVATION_PATH)

# Backup original get_activation_fn
orig_get_activation_fn = utils.get_activation_fn
# ────────────────────────────────────────────────────────────────────────────────────────────────
def custom_get_activation_fn(activation: str):
    activation = activation.lower()
    if activation == "tanhexp":
        from activation.TanhExp import TanhExp
        print("✅ Registered custom activation: TanhExp")
        return TanhExp()
    
    elif activation == "fftgate":
        from activation.FFTGate import FFTGate
        print("✅ Registered custom activation: FFTGate")
        return FFTGate()
    
    elif activation == "geglu":        
        from activation.GEGLU import GEGLU
        print("✅ Registered custom activation: GEGLU")
        return GEGLU()

    else:
        # ✅ fallback to original fairseq activations
        return orig_get_activation_fn(activation)
# ────────────────────────────────────────────────────────────────────────────────────────────────
# Patch fairseq utils
utils.get_activation_fn = custom_get_activation_fn

# ────────────────────────────────────────────────────────────────────────────────────────────────
print(f"⚡ Using activation function: {exp_args.act_name.lower()}")

# Test-call the patched activation to confirm it works
try:
    act_fn = utils.get_activation_fn(exp_args.act_name.lower())
    print(f"🔍 Activation function resolved: {act_fn}")
except Exception as e:
    print(f"❌ Failed to resolve activation '{exp_args.act_name.lower()}': {e}")
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Manually inject custom activation into config
cfg.model.activation_fn = exp_args.act_name.lower()
print(f"⚡ Overwrote cfg.model.activation_fn = {cfg.model.activation_fn}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  3️⃣ ========  4.3 Fairseq Task & Dataset Initialization ========================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅✅ Load dictionaries
src_dict = Dictionary.load(os.path.join(fairseq_args.data, f"dict.{fairseq_args.source_lang}.txt"))
tgt_dict = Dictionary.load(os.path.join(fairseq_args.data, f"dict.{fairseq_args.target_lang}.txt"))

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅✅ Setup task
task = TranslationTask.setup_task(cfg.task)
task.load_dataset("train")
task.load_dataset("test")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ✅✅ Print translation direction (ZH→EN confirmation)
print(f"🌍 Translation direction: {fairseq_args.source_lang} → {fairseq_args.target_lang}")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ✅✅ Inspect dataset lengths after loading
train_dataset = task.dataset("train")
test_dataset  = task.dataset("test")
# ────────────────────────────────────────────────────────────────────────────────────────────────

print(f"📏 Train set - Max source length: {train_dataset.src_sizes.max()}, Max target length: {train_dataset.tgt_sizes.max()}")
print(f"📏 Test set  - Max source length: {test_dataset.src_sizes.max()}, Max target length: {test_dataset.tgt_sizes.max()}")

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅✅ Inspect data you’d lose if you lower max_positions
def check_percentage(dataset, N):
    src_long = (dataset.src_sizes > N).sum()
    tgt_long = (dataset.tgt_sizes > N).sum()
    total = len(dataset)

    print(f"Dataset size: {total}")
    print(f"  > {N} tokens (source): {src_long} ({100*src_long/total:.2f}%)")
    print(f"  > {N} tokens (target): {tgt_long} ({100*tgt_long/total:.2f}%)")
    print()

# ✅✅ Check on train + test
check_percentage(train_dataset, exp_args.max_source_positions)
check_percentage(test_dataset, exp_args.max_source_positions)
# ─────────────────────────────────────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 5. MODEL INITIALIZATION | XXX ----------------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 5. MODEL INITIALIZATION | XXX ----------------------------------------------------####################
########################################################################################################################

# ================================================================================================
# 🏷️========================= MODEL INITIALIZATION===============================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####


# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  ======== 🔖🔖  5.1 Model Construction 🔖🔖====================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 


# ================================================================================================
# 1️⃣ 🧠 Build Fairseq Baseline Transformer
# ================================================================================================
# ================================================================================================
# 1️⃣ 🧠 Build Fairseq Baseline Transformer
# ================================================================================================

if exp_args.net == "Transformer":

    model = task.build_model(cfg.model)

    print(f"✅ Loaded Model: {model.__class__.__name__}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ================================================================================================
# 2️⃣ 🧠 Build TDM_Former 📐🔑🏷️
# ================================================================================================
# ================================================================================================
# 2️⃣ 🧠 Build TDM_Former 📐🔑🏷️
# ================================================================================================
elif exp_args.net == "TDM_Former":


    tdm_layers = []

    # ───────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ 🧩🧠 === FFN Insertion === ♻️♻️ ========================================================== 
    # ───────────────────────────────────────────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ 🧩🧠 === FFN Insertion === ♻️♻️ ========================================================== 
    # ───────────────────────────────────────────────────────────────────────────────────────────────
    if exp_args.tdm_insertion_type == "ffn":

        # ================================================================================================
        # 1️⃣.0️⃣ 🧠 Build TDM_Former  
        # ================================================================================================
        from models.TDM_Former import TDM_Former
        print(f"♻️♻️ Loaded Model: TDM_Former for insertion_typ:ffn🧬🧬")
        print(f"♻️♻️ Loaded Model: TDM_Former for insertion_typ:ffn🧬🧬")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────

        # # 🔒 IMPORTANT: reset RNG after importing TDM_Former, before the real training model is built (For Repeatable Initialization)
        # set_seed_torch(seed1)
        # set_seed_main(seed2)

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # =============================================================================
        # 2️⃣.1️⃣📌📌 Load baseline Fairseq Transformer 🅰️🔼 
        # =============================================================================
        model = task.build_model(cfg.model)

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        print("\nMODEL INIT CHECK")
        print(model.decoder.layers[0].fc1.weight[0, :10])
        print()
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────────────────────────────────       
        # =============================================================================
        # 1️⃣.2️⃣📌📌 Inject/Apply TDM enhancement to decoder FFNs 🅱️🔼
        # =============================================================================
        for i, layer in enumerate(model.decoder.layers):

            if i >= len(model.decoder.layers) - exp_args.tdm_num_layers:

                original_activation_fn = layer.activation_fn

                tdm = TDM_Former(
                    dim=layer.fc1.out_features,
                    alpha=exp_args.tdm_alpha,
                    amplification=exp_args.tdm_transition_amplification,
                    tdm_transition=exp_args.tdm_transition,
                    tdm_rotation=exp_args.tdm_rotation                    
                )
                #--------------------------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------------------------
                # 🧩🔒 Register TDM Module (Architecture Visibility)
                #--------------------------------------------------------------------------------------------------
                if exp_args.tdm_register_module:

                    # 🔒 Configure TDM coefficient settings
                    tdm.alpha.requires_grad = False
                    tdm.amplification.requires_grad = False

                    layer.tdm_module = tdm

                #--------------------------------------------------------------------------------------------------
                # 🔗🔒 Closure-Based TDM Injection (Activation Wrapper Only)
                #--------------------------------------------------------------------------------------------------
                else:

                    pass
                # ───────────────────────────────────────────────────────────────────────────────────────────────── 
                # ─────────────────────────────────────────────────────────────────────────────────────────────────           

                # ─────────────────────────────────────────────────────────────────────────────────────────────────
                def wrapped_activation(
                    x,
                    act_fn=original_activation_fn,
                    tdm_module=tdm
                ):

                    # ==========================================================
                    # 1️⃣.2️⃣.1️⃣ Standard FFN activation
                    # ==========================================================
                    x = act_fn(x)

                    # ==========================================================
                    # 1️⃣.2️⃣.2️⃣ Previous hidden state
                    # ==========================================================
                    prev_x = torch.roll(
                        x,
                        shifts=1,
                        dims=0
                    )

                    prev_x[0] = x[0]

                    # ======================================================================
                    # 🧬🔑 1️⃣.2️⃣.3️⃣ CORE TDM: Transition dynamics extraction 🧬🔑
                    # ----------------------------------------------------------------------
                    # Computes hidden-state evolution: Δ_t = h_t - h_{t-1}    
                    # ----------------------------------------------------------------------                
                    # ======================================================================
                    if tdm_module.tdm_transition:                    
                        transition = x - prev_x

                        transition = transition / (
                            1e-6 +
                            transition.abs().mean(
                                dim=-1,
                                keepdim=True
                            )
                        )

                    else:

                        transition = x

                    # ======================================================================
                    # 🧬🔑 1️⃣.2️⃣.4️⃣ CORE TDM: Transition-conditioned response 🧬🔑
                    # ----------------------------------------------------------------------
                    # Converts transition dynamics into adaptive modulation signal
                    # ----------------------------------------------------------------------
                    # ======================================================================
                    transition_scale = torch.sigmoid(
                        tdm_module.alpha * torch.tanh(
                            transition
                        )
                    )

                    # ======================================================================
                    # 🧬🔑 1️⃣.2️⃣.5️⃣ CORE TDM: Transition-conditioned FFN modulation 🧬🔑
                    # ----------------------------------------------------------------------
                    # Modulates FFN activation using transition dynamics
                    # ----------------------------------------------------------------------
                    # ======================================================================

                    global TDM_SCALE

                    x_base = x

                    x_mod = x * (
                        1.0 + tdm_module.amplification * transition_scale
                    )
                    #-----------------------------------------------------------
                    x_mod = tdm_module(x_mod)

                    # ==========================================================
                    # 1️⃣.2️⃣.6️⃣ Residual TDM integration
                    # ==========================================================
                    x = x_base + TDM_SCALE * (x_mod - x_base)

                    return x
               # ─────────────────────────────────────────────────────────────────────────────────────────────────
               # ─────────────────────────────────────────────────────────────────────────────────────────────────
               

                # =============================================================================
                # 📌📌 Replace decoder activation function
                # =============================================================================                 
                layer.activation_fn = wrapped_activation
                # ─────────────────────────────────────────────────────────────────────────────────────────────────  
                # ─────────────────────────────────────────────────────────────────────────────────────────────────  


                # ==========================================================
                # 🚦🔍 TDM inserted into decoder layer 
                # ==========================================================                  
                tdm_layers.append(i)
                print(f"📣 FFN Insertion | TDM inserted into decoder layer {i}")                
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        print(f"✅ Loaded Model: TDM_Former")
        print(f"✅ Loaded Model: {model.__class__.__name__}")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(f"Unknown TDM insertion type: {exp_args.tdm_insertion_type}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    print(f"📣📣 TDM_Former for insertion_typ={exp_args.tdm_insertion_type} | Loaded Model={exp_args.net }♻️♻️")
    print(f"📣📣 TDM_Former for insertion_typ={exp_args.tdm_insertion_type} | Loaded Model={exp_args.net }♻️♻️")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    #  ======== 🔖  5.1.1 Model Configuration Hyperparameter Summary ==================================
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    if exp_args.net == "TDM_Former":

        tdm_log = (
            f"📣 TDM Configuration             : "
            f"alpha={exp_args.tdm_alpha} | "
            f"amp={exp_args.tdm_transition_amplification} | "
            f"layers={exp_args.tdm_num_layers} | "
            f"insertion={exp_args.tdm_insertion_type} | "
            f"transition={exp_args.tdm_transition} | "
            f"rotation={exp_args.tdm_rotation} | "
            f"scale_init={exp_args.tdm_scale} | "
            f"start_epoch={exp_args.tdm_start_epoch} | "
            f"full_epoch={exp_args.tdm_full_epoch} "
        )

        print(tdm_log)

    else:

        print(
            f"📣 Baseline Configuration        : "
            f"model={exp_args.net}"
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ======== 🔖🔖  5.2  Insert model to device 🔖🔖================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

model = model.to(DEVICE)
 # ─────────────────────────────────────────────────────────────────────────────────────────────────
print(f"🚀🚀 Model moved to device: {DEVICE} ♻️♻️")
print(f"🚀🚀 Model moved to device: {DEVICE} ♻️♻️")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ======== 🔖🔖 5.3  Model Validation & Sanity Checks 🔖🔖=======================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔍 Print custom activations to confirm they are used
for name, module in model.named_modules():
    if isinstance(module, nn.Module) and module.__class__.__name__ in ["TanhExp", "FFTGate"]:
        print(f"🔎 Found custom activation in {name}: {module}")


# ✅ Sanity check Device
if DEVICE.type == "cuda":
    print(f"🚀 Model moved to CUDA: {torch.cuda.get_device_name(DEVICE)}")
else:
    print("⚠️ Using CPU (CUDA not available)")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 6. PATH DEFINATION | XXX --------------------------------------------------------#####################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ 📌📌 ======== ENSURE DIRECTORY EXIST ========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# # 🟡 === Checkpoint directories ===
# if not os.path.exists('checkpoint'):
#     os.makedirs('checkpoint')
# # ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Results directories ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
if not os.path.exists(f'./Results/{exp_args.dataset_name}'):
    os.makedirs(f'./Results/{exp_args.dataset_name}')

if not os.path.exists(f'./Results/{exp_args.dataset_name}/{exp_args.net}'):
    os.makedirs(f'./Results/{exp_args.dataset_name}/{exp_args.net}')

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === LR Log directory ===
# ────────────────────────────────────────────────────────────────────────────────────────────────
if not os.path.exists(f'./Results/{exp_args.dataset_name}/{exp_args.net}'):
    os.makedirs(f'./Results/{exp_args.dataset_name}/{exp_args.net}')

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Training Log directory ===
# ────────────────────────────────────────────────────────────────────────────────────────────────
if not os.path.exists(f'./Results/{exp_args.dataset_name}/{exp_args.net}'):
    os.makedirs(f'./Results/{exp_args.dataset_name}/{exp_args.net}')

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Model Summary directory ===
# ────────────────────────────────────────────────────────────────────────────────────────────────
if not os.path.exists(f'./Results/{exp_args.dataset_name}/{exp_args.net}'):
    os.makedirs(f'./Results/{exp_args.dataset_name}/{exp_args.net}')
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ 📌📌 ======== DEFINE DIRECTORY ==============================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣.1️⃣🔖 ==== Checkpoint Directory Selection | Normal Run vs Crash Run | save every epoch ===== 
# ────────────────────────────────────────────────────────────────────────────────────────────────

# 🅰️💾 Normal run: save checkpoints in standard folder
if not exp_args.crash_run:
    checkpoint_dir = f'./checkpoints/{exp_args.dataset_name}/{exp_args.net}/'

    # 📁 Create checkpoint folder if it does not exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"🅰️💾 Normal Run Enabled | Saving checkpoints to: {checkpoint_dir}")
#---------------------------------------------------------------------------------------------------
# 🅱️♻️ Crash run: save checkpoints in recovery folder
else:
    checkpoint_dir = f'./checkpoints_crash_run/{exp_args.dataset_name}/{exp_args.net}/'

    # 📁 Create checkpoint folder if it does not exist    
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"🅱️♻️ Crash Run Enabled | Saving recovery checkpoints to: {checkpoint_dir}")

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣.2️⃣🔖 === Main Test & Train Results === 
train_results_path = f'./Results/{exp_args.dataset_name}/{exp_args.net}/Train_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
test_results_path = f'./Results/{exp_args.dataset_name}/{exp_args.net}/Test_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣.3️⃣🔖  === LR, Training & Summary logs  === 
LR_save_paths = {
    "LR_history": f"./Results/{exp_args.dataset_name}/{exp_args.net}/{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_LR_log.txt"
}
save_paths = {
    "log_history": f"./Results/{exp_args.dataset_name}/{exp_args.net}/{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_training_logs.txt"
}
# ────────────────────────────────────────────────────────────────────────────────────────────────

# 2️⃣.4️⃣🔖 ===  Model summary path === 
model_summary_path = f'./Results/{exp_args.dataset_name}/{exp_args.net}/{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_model_summary.txt'
model_summary_full_path = f'./Results/{exp_args.dataset_name}/{exp_args.net}/{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_model_summary_full.txt'
# ────────────────────────────────────────────────────────────────────────────────────────────────

# # ✅🔖 === Path template for saving checkpoints at every epoch | Part2 (if scratch occured) ===
# checkpoint_dir = f'./checkpoints/{exp_args.dataset_name}/tanhexp_AfterCrash_41_49epochs/'
# os.makedirs(checkpoint_dir, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 7. TRAINING INITIALIZATION & OPTIMIZATION SETUP | XXX ---------------------------#####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  1️⃣ ========  7.1 AMP GradScaler ===============================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Initialize AMP GradScaler once globall
scaler = torch.cuda.amp.GradScaler()
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣ ========  Training Criterion ===============================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

criterion = LabelSmoothingCrossEntropy(smoothing=exp_args.smoothing)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  3️⃣ ========   Optimizer Configuration =========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ♻️ Adam optimizer (β1=0.9, β2=0.98, eps=1e-9) ♻️
# ─────────────────────────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=exp_args.lr,
    betas=(0.9, 0.98),
    eps=exp_args.eps,
    weight_decay=exp_args.weight_decay,
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣ ========  Learning Rate Scheduler ===========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ✅  Transformer LR schedule (warmup + inverse sqrt decay)
# --------------------------

warmup_updates = exp_args.warmup_updates   # configurable warmup steps
peak_lr = exp_args.lr

# ────────────────────────────────────────────────────────────────────────────────────────────────
def lr_lambda(step):
    step = max(1, step)
    if step == 1:
        print(f"🚀 Warmup started at step {step} (target peak LR = {peak_lr}).")
    if step == warmup_updates:
        actual_lr = optimizer.param_groups[0]["lr"]
        print(f"✅ Warmup finished at step {step}. Peak LR reached: {actual_lr:.8f}")
    if step == warmup_updates + 1:
        print(f"📉 Inverse square root decay started at step {step}.")
    
    if step <= warmup_updates:
        # Linear warmup
        scale = step / warmup_updates
    else:
        # Inverse square root decay
        scale = (warmup_updates ** 0.5) / (step ** 0.5)
    return scale

# ────────────────────────────────────────────────────────────────────────────────────────────────

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣ ========  Global Tracking Variables ===========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ global initialization
best_bleu = 0.0
total_test_duration = 0.0

# ♻️ Initialize global TDM scale before any forward call
TDM_SCALE = exp_args.tdm_scale
TDM_SCALE_LOG = ""
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# %%




########################################################################################################################
####-------| NOTE 8. TRAIN FUNCTIONS | XXX --------------------------------------------------------#####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 8. TRAIN FUNCTIONS | XXX --------------------------------------------------------#####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Training  ===
def train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler):
    epoch_start_time = time.time()
    model.train()


    # ──────────────────────────────────
    # ❌ GPU memory check at epoch start
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} TRAIN START] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    train_iter = task.get_batch_iterator(
        dataset=task.dataset("train"),
        max_tokens=fairseq_args.max_tokens,
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        seed=seed1,       
        num_shards=exp_args.num_shards,                 
        shard_id=exp_args.shard_id,                     
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs
    ).next_epoch_itr(shuffle=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    epoch_loss = 0.0
    # ✅ Initialize LR log list for per-batch LR tracking
    lr_log_history = []


    # ✅ Initialize log history
    log_history = []
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    with tqdm(enumerate(train_iter), total=len(train_iter), desc=f"Train Epoch {epoch}") as progress:
        for i, sample in progress:
            # ────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Move entire sample dict to DEVICE
            sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample

            # ✅ Compute step
            step_idx = epoch * len(train_iter) + i

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug LR
            lr_now = optimizer.param_groups[0]["lr"]
            wd_now = optimizer.param_groups[0]["weight_decay"]

            detailed_steps = {0, 1, 2, 5, 10, 50, 70, 100, 200, 500, 1000, 1500, 1700, 3000, 4000, 5000}
            detailed_steps.add(len(train_iter) - 1)
            milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 40, 45}

            if (epoch in milestone_epochs) and (i in detailed_steps):
                lr_log_msg = (
                    f"[Epoch {epoch} | Batch {i} | Global Step {step_idx}] "
                    f"Main LR: {lr_now:.8f} | WD: {wd_now:.8f}"
                )
                lr_log_history.append(lr_log_msg)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            src_tokens = sample["net_input"]["src_tokens"]
            prev_output_tokens = sample["net_input"]["prev_output_tokens"]
            target = sample["target"]

            src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)

            optimizer.zero_grad()
            # ────────────────────────────────────────────────────────────────────────────────────────────────

            # # ✅ Forward pass for loss without AMP
            # logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ ⚠️ Forward + loss under autocast (AMP) with safe try/except 
            try:    
                with torch.cuda.amp.autocast():   # AMP context
                    logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
                    # ────────────────────────────────────────────────────────────────────────────────────────────────
                    # ✅ with (pad-masked)
                    pad = tgt_dict.pad()
                    logits2d = logits.view(-1, logits.size(-1))
                    target1d = target.view(-1)
                    mask = target1d != pad
                    if mask.any():
                        loss = criterion(logits2d[mask], target1d[mask])
                    else:
                        loss = logits2d.sum() * 0.0  # degenerate case safeguard

                    # ────────────────────────────────────────────────────────────────────────────────────────────────
                    # ✅ Check for NaN/Inf loss
                    if not torch.isfinite(loss):
                        print(f"💥⚠️ Training Loop: Non-finite loss (NaN/Inf) at epoch {epoch}, batch {i}. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue   # Skip this batch safely

            except RuntimeError as e:  
                print(f"💥 ⚠️ Training Loop: AMP forward failed at epoch {epoch}, batch {i}: {e}")  # AMP failure
                torch.cuda.empty_cache()
                continue   # Skip this batch and move on safely
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # loss.backward()
            # optimizer.step()

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ ⚠️ Backward pass with gradient scaler (AMP)
            scaler.scale(loss).backward()
            scaler.step(optimizer)       # replaces optimizer.step()
            scaler.update()              # unscales gradients & adjusts scaling
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ────────────────────────────────────────────────────────────────────────────────────────────────
            lr_scheduler.step()  # ✅ update LR per batch (Transformer schedule) NOt affected by AMP

            epoch_loss += loss.item()
            # ────────────────────────────────────────────────────────────────────────────────────────────────

            # ────────────────────────────────────────────────────────────────────────────────────────────────
            progress.set_postfix(
                loss=round(epoch_loss / (i + 1), 3),
                lr=f"{lr_now:.8f}",
                wd=f"{wd_now:.8f}",
            )
            # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    avg_loss = epoch_loss / len(train_iter)

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Timing
    duration = time.time() - epoch_start_time
    mins, secs = divmod(duration, 60)
    print(f"⏱ Epoch {epoch} Training time {exp_args.net}: {int(mins)} min {secs:.2f} sec")
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ after loop finishes (end of epoch)
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    log_msg = (
        f"Epoch {epoch} | Last Batch {i} | "
        f"M_Optimizer LR => {current_lr:.8f} | WD => {current_wd:.6f} | "
        f"⏱ Training time => {exp_args.net}: {int(mins)} min {secs:.2f} sec"
    )
    log_history.append(log_msg)
    print(log_msg)
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # # ────────────────────────────────────────────────────────────────────────────────────────────────
    # # ✅ Save logs | Clear the log file at the start of training (Epoch 0)
    # if epoch == exp_args.start_epoch:
    #     os.makedirs(os.path.dirname(train_results_path), exist_ok=True)
    #     with open(train_results_path, "w", encoding="utf-8") as f:
    #         f.write("")  # ✅ Clears previous logs only once at the start

    # with open(train_results_path, "a", encoding="utf-8") as f:
    #     f.write(f"Epoch {epoch} | Train Loss: {avg_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.8f}\n")
    # # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒📦 Save logs | Clear the log file at the start of training (Epoch 0) 📦🔒
    # ────────────────────────────────────────────────────────────────────────────────────────────────    
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒📦 Save logs | Clear the log file at the start of training (Epoch 0) 📦🔒
    # ────────────────────────────────────────────────────────────────────────────────────────────────    
    if epoch == exp_args.start_epoch:
        os.makedirs(os.path.dirname(train_results_path), exist_ok=True)
        with open(train_results_path, "w", encoding="utf-8") as f:
            f.write("")  # ✅ Clears previous logs only once at the start

            # ---------------------------------------------------------------------------
            # 1️⃣🏷️ Training Model Status
            # ---------------------------------------------------------------------------
            if exp_args.net == "TDM_Former":

                f.write(
                    f"Model: TDM_Former\n"
                    f"Ablation Status: "
                    f"transition={exp_args.tdm_transition} | "
                    f"rotation={exp_args.tdm_rotation} | "
                    # f"tdm_layers={exp_args.tdm_num_layers} | "

                    f"enhanced_layers=Last {exp_args.tdm_num_layers} decoder FFN layers "
                    f"(L{6-exp_args.tdm_num_layers+1}-L6) | "

                    f"insertion={exp_args.tdm_insertion_type}\n\n"
                )

            else:

                f.write(
                    # f"Model: Baseline Fairseq Transformer\n\n"
                    f"Model: Baseline Fairseq Transformer\n"
                    f"Ablation Status: transition=False | rotation=False | enhanced_layers=None | insertion=None\n\n"                    
                )

    # ---------------------------------------------------------------------------
    # 2️⃣🏷️ Append epoch training results
    # --------------------------------------------------------------------------- 
    with open(train_results_path, "a", encoding="utf-8") as f:
        # f.write(f"Epoch {epoch} | Train Loss: {avg_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.8f}\n")
        f.write(
            f"Epoch {epoch} "
            f"| Train Loss: {avg_loss:.3f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.8f}"
            f"{TDM_SCALE_LOG}\n"
        )        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────          
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Initialize log file at the beginning of training (Clear old logs)
    if epoch == exp_args.start_epoch:  # ✅ Only clear at the start of training
        with open(save_paths["log_history"], "w", encoding="utf-8") as log_file:
            log_file.write("")  # ✅ Clears previous logs

    # ✅ Save logs once per epoch (Append new logs)
    if log_history:
        with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_history) + "\n")          # ✅ Ensure each entry is on a new line
        print(f"📜 Logs saved to {save_paths['log_history']}!")    # ✅ Only prints once per epoch
    else:
        print("⚠ No logs to save!")
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save LR log history to file (once per epoch)
    if epoch == exp_args.start_epoch:
        os.makedirs(os.path.dirname(LR_save_paths["LR_history"]), exist_ok=True)
        with open(LR_save_paths["LR_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch

    if lr_log_history:
        os.makedirs(os.path.dirname(LR_save_paths["LR_history"]), exist_ok=True)
        with open(LR_save_paths["LR_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(lr_log_history) + "\n")
    #     print(f"📈 LR logs saved to {LR_save_paths['LR_history']}!")
    # else:
    #     print("⚠ No LR logs to save.")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    print(f"📊 Train Loss: {avg_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"📜 Training logs saved to {train_results_path}!")


    # ──────────────────────────────────
    # ❌ GPU memory check at At the end of each epoch (training)
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} AFTER TRAIN] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save checkpoint at the end of each epoch
    ckpt_path = os.path.join(checkpoint_dir,
        f"epoch_{epoch}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.mode_name}.pt"
    )
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "train_loss": avg_loss
    }, ckpt_path)
    print(f"🟡🟡 Epoch {epoch} checkpoint saved to: {ckpt_path}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    return avg_loss
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 9. TEST FUNCTIONS (with sacreBLEU) | BEST PRACTICE --------------------------------------------------#
########################################################################################################################

# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# #  1️⃣ ========  9.1  BLEU Detokenization (🔖 EN/FR/DE/IT/RO/NL) ==================================
# # ───────────────────────────────────────────────────────────────────────────────────────────────── 
# # ────────────────────────────────────────────────────────────────────────────────────────────────
# def detokenize_for_bleu(s: str) -> str:
#     s = s.replace("@@ ", "").replace("@@", "")  # BPE
#     s = s.replace("▁", " ")                     # SentencePiece
#     s = " ".join(s.split())                     # normalize spaces
#     s = re.sub(r"\s+([?.!,])", r"\1", s)        # punctuation
#     return s.strip()
# # ────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  1️⃣ ========  9.1  BLEU Detokenization (🔖 Chinese) ============================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
def detokenize_for_bleu(s: str) -> str:
    # For Chinese, don't remove SentencePiece boundaries or spaces | Only remove BPE continuation markers
    # sacreBLEU's "zh" tokenizer expects raw characters.
    return s.replace("@@ ", "").replace("@@", "").strip()
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣ ========  9.2 Testing Loop ==================================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Testing  ===
def test_one_epoch(epoch, task, model, criterion, beam_size=None):

    global best_bleu, total_test_duration

    # ──────────────────────────────────
    # ❌ GPU memory check At test start
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} TEST START] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────

    model.eval()



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Build test iterator first
    test_iter = task.get_batch_iterator(
        dataset=task.dataset("test"),
        max_tokens=fairseq_args.max_tokens,  # cut in half option: fairseq_args.max_tokens // 2
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        seed=seed1,
        num_shards=exp_args.num_shards,
        shard_id=exp_args.shard_id,
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs,
            ).next_epoch_itr(shuffle=False)


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ References
    true_references = []
    for sample in test_iter:
        targets = sample["target"]
        for t in targets:
            t_no_pad = t[t != tgt_dict.pad()]  # remove pads
            ref_str = tgt_dict.string(t_no_pad, bpe_symbol="@@")
            true_references.append(detokenize_for_bleu(ref_str))
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ❗ Rebuild test_iter again (since the first one is exhausted after above loop)
    test_iter = task.get_batch_iterator(
        dataset=task.dataset("test"),
        max_tokens=fairseq_args.max_tokens,
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        seed=seed1,
        num_shards=exp_args.num_shards,
        shard_id=exp_args.shard_id,
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs,     
    ).next_epoch_itr(shuffle=False)
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ ===== Build a LOCAL generation config =====
    gen_cfg = deepcopy(cfg.generation)
    if beam_size is not None:
        gen_cfg.beam = int(beam_size)
    # else: keep whatever was set in --eval-bleu-args (beam=5, max_len_a=1.2, max_len_b=10)

    gen_cfg.lenpen = 1.0   # ✅ Fairseq default (neutral length penalty: no bias for longer/shorter translations)

    generator = task.build_generator([model], gen_cfg)
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    test_loss = 0.0
    sys_outputs = []
    start_time = time.time()
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        with tqdm(enumerate(test_iter), total=len(test_iter), desc=f"Test Epoch {epoch}") as progress:
            for i, sample in progress:
                # ✅ Move the whole sample to device (safe on CPU/GPU)
                sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample
                # ────────────────────────────────────────────────────────────────────────────────────────────────
                src_tokens = sample["net_input"]["src_tokens"]
                prev_output_tokens = sample["net_input"]["prev_output_tokens"]
                target = sample["target"]
                src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)
                # ────────────────────────────────────────────────────────────────────────────────────────────────

                # # ✅ Forward pass for loss without AMP
                # logits, _ = model(src_tokens, src_lengths, prev_output_tokens)



                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ ⚠️ Forward + loss under autocast (AMP) with safe try/except
                try:
                    with torch.cuda.amp.autocast():   # AMP context
                        logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
                        # ────────────────────────────────────────────────────────────────────────────────────────────────
                        pad = tgt_dict.pad()
                        logits2d = logits.view(-1, logits.size(-1))
                        target1d = target.view(-1)
                        mask = target1d != pad
                        loss = criterion(logits2d[mask], target1d[mask]) if mask.any() else logits2d.sum() * 0.0
                        # ────────────────────────────────────────────────────────────────────────────────────────────────

                        # ✅ Check for NaN/Inf loss
                        if not torch.isfinite(loss):
                            print(f"💥⚠️ Test Loop: Non-finite loss (NaN/Inf) at epoch {epoch}, batch {i}. Skipping batch.")
                            torch.cuda.empty_cache()
                            continue           # Skip this batch safely
                # ────────────────────────────────────────────────────────────────────────────────────────────────                
                # ✅ Check for Runtime Error
                except RuntimeError as e:  
                    print(f"💥 ⚠️ Test Loop: AMP forward failed at epoch {epoch}, batch {i}: {e}")  # AMP failure
                    torch.cuda.empty_cache()
                    continue                   # Skip this batch and move on safely
                # ────────────────────────────────────────────────────────────────────────────────────────────────



                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ Compute Loss
                test_loss += loss.item()
                # ────────────────────────────────────────────────────────────────────────────────────────────────



                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ Proper decoding via generator (uses local beam_size)
                hypos = task.inference_step(generator, [model], sample)

                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ System outputs
                for j, hypos_j in enumerate(hypos):
                    hypo_tokens = [tok for tok in hypos_j[0]["tokens"].tolist() if tok != tgt_dict.pad()]  # remove pads
                    sys_str = tgt_dict.string(hypo_tokens, bpe_symbol="@@")
                    sys_outputs.append(detokenize_for_bleu(sys_str))

                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # # 🔍 also log the aligned source sentence for debugging (only once for very first example overall)
                # if len(sys_outputs) == 1:
                #     src_str = src_dict.string(sample["net_input"]["src_tokens"][j], bpe_symbol="@@")
                #     print("----- DEBUG OUTPUT (FIRST EXAMPLE) -----")
                #     print("SRC:", detokenize_for_bleu(src_str))
                #     print("SYS:", sys_outputs[0], "| REF:", true_references[0])
                #     print("---------------------------------------")
                # ────────────────────────────────────────────────────────────────────────────────────────────────

                progress.set_postfix(loss=round(test_loss / (i + 1), 3))
                # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Compute Ave Loss
    avg_loss = test_loss / max(1, len(test_iter))
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # # 🔍 Debug: check alignment again at summary level
    # print("----- DEBUG OUTPUT -----")
    # print("SYS[0]:", sys_outputs[0][:200])   # first hypothesis (truncated to 200 chars)
    # print("REF[0]:", true_references[0][:200])  # first reference
    # print("SYS length:", len(sys_outputs), "| REF length:", len(true_references))


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅  ❗ sacreBLEU computation
    beam_used = beam_size if beam_size is not None else cfg.generation.beam
    
    # bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references])                 #🔖 EN/FR/DE/IT/RO/NL
    bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references], tokenize="zh")    #🔖 Chinese


    print(f"🌍 sacreBLEU (beam={beam_used}): {bleu.score:.2f}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    duration = time.time() - start_time
    total_test_duration += duration
    mins, secs = divmod(duration, 60)
    print(f"⏱ Test Epoch {epoch}: {int(mins)}m {secs:.1f}s | Loss {avg_loss:.3f} | sacreBLEU {bleu.score:.2f}")


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save best checkpoint
    if bleu.score > best_bleu:
        best_bleu = bleu.score
        ckpt_path = (
            f'./checkpoints/{exp_args.dataset_name}/{exp_args.net}/{exp_args.net}_{exp_args.dataset_name}'
            f'_LR{str(exp_args.lr).replace(".", "_")}_{exp_args.optimizer1}_{exp_args.mode_name}_LastEpoch{epoch}.pt'
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)   # ✅ FIX HERE
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "bleu": bleu.score,
        }, ckpt_path)
        print(f"🏆 New best checkpoint saved: {ckpt_path}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Log results (ensure directory exists)
    # if epoch == exp_args.start_epoch:        # Refresh at start_epoch
    if epoch == (exp_args.epochs - 1):       # Refresh at End_epoch
        os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
        if os.path.exists(test_results_path):
            open(test_results_path, "w").close()

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    beam_used = beam_size if beam_size is not None else cfg.generation.beam
    with open(test_results_path, "a", encoding="utf-8") as f:
        f.write(
            f"Epoch {epoch} | Test Loss: {avg_loss:.3f} "
            f"| sacreBLEU: {bleu.score:.2f} "
            f"| Beam={beam_used}\n"
        )
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Append the Last BLEU score only once at the end of training
    if epoch == (exp_args.epochs - 1):  
        with open(test_results_path, "a", encoding="utf-8") as f:
            f.write(f"\n🏆 Last BLEU Score: {best_bleu:.2f}\n")

            # 🟡 Log total test time at the END
            total_mins, total_secs = divmod(total_test_duration, 60)
            f.write(f"\n🕒 Total Test Time => {exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")
            print(f"🕒 Total Test Time => {exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec")
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Print Final BLEU
    print(f"📊 Test BLEU: {bleu.score:.2f} | 🏆 Final BLEU: {best_bleu:.2f}")
    print(f"📜 Test logs saved to {test_results_path}!")
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    return avg_loss, bleu.score
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────



########################################################################################################################
####-------| NOTE 10. MAIN LOOP | XXX ----------------------------------------------------------########################
########################################################################################################################
########################################################################################################################
####-------| NOTE 10. MAIN LOOP | XXX ----------------------------------------------------------########################
########################################################################################################################

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    training_total_start = time.time()
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    set_seed_torch(seed1)
    set_seed_main(seed2)
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Free unused GPU memory BEFORE training starts
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 1️⃣ 🧩🧠 === Baseline Fairseq ♻️ ♻️ ========================================================== 
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    if exp_args.net == "Transformer":    
        for epoch in range(exp_args.start_epoch, exp_args.epochs):

            # ───────────────────────────────────────────────────────────────────────────────
            # 🤖📚 Baseline Fairseq Transformer Training 🏗️📈
            # ───────────────────────────────────────────────────────────────────────────────
            train_loss = train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler)
            tqdm.write("")
            # ────────────────────────────────────────────────────────────────────────────────────────────────
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ 🧩🧠 === TDM_Former ♻️ ♻️ ================================================================ 
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif exp_args.net == "TDM_Former":
        import math

        TDM_START_EPOCH = exp_args.tdm_start_epoch
        TDM_FULL_EPOCH = exp_args.tdm_full_epoch

        for epoch in range(exp_args.start_epoch, exp_args.epochs):

            if epoch < TDM_START_EPOCH:
                TDM_SCALE = exp_args.tdm_scale
            else:
                progress = min(
                    1.0,
                    (epoch - TDM_START_EPOCH) / (TDM_FULL_EPOCH - TDM_START_EPOCH)
                )

                TDM_SCALE = 0.5 * (1.0 - math.cos(math.pi * progress))

            print(f"📐 TDM_SCALE for epoch {epoch}: {TDM_SCALE:.3f}")
            #------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------

            # ───────────────────────────────────────────────────────────────────────────────
            # 🤖🧠 TDM_Former Training | Transition Dynamics Modulation 🧬📈
            # ───────────────────────────────────────────────────────────────────────────────            
            train_loss = train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler)
            tqdm.write("")
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(f"Unknown model type: {exp_args.net}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # # ✅ Conditional GPU cleanup before Test
    # mem_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
    # total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # if mem_reserved > 0.8 * total_mem:
    #     print(f"⚠️ BEFORE TEST: High GPU reserved memory ({mem_reserved:.2f} GB / {total_mem:.2f} GB). Cleaning cache...")
    #     torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Test 
    test_loss, bleu_score_val = test_one_epoch(epoch, task, model, criterion)
    tqdm.write("")

    print("Final BLEU: ", best_bleu)

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    training_total_end = time.time()
    total_mins, total_secs = divmod(training_total_end - training_total_start, 60)
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_paths["log_history"]), exist_ok=True)
    with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
        log_file.write(f"\n🕒 Total Training Time => {exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")

    os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
    with open(test_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n🕒 Total Training Time => {exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")

    print(f"🕒 Total Training Time: {int(total_mins)}m {total_secs:.1f}s")
    # ────────────────────────────────────────────────────────────────────────────────────────────────




# %%

########################################################################################################################
####-------| NOTE 11. TEST ALL CHECKPOINTS (BLEU per epoch)  | XXX -----------------------------########################
########################################################################################################################
########################################################################################################################
####-------| NOTE 11. TEST ALL CHECKPOINTS (BLEU per epoch)  | XXX -----------------------------########################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  1️⃣ ======== Paths for checkpoint BLEU sweeps ===================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Paths for checkpoint BLEU sweeps
test_checkpoint_results_path = (
    f'./Results/{exp_args.dataset_name}/{exp_args.net}/'
    f'TestCheckpoints_{exp_args.net}_'
    f'{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
)
# ────────────────────────────────────────────────────────────────────────────────────────────────


# # ✅ Paths for checkpoint BLEU sweeps | Part2 (if Cratch occured)
# test_checkpoint_results_path = (
#     f'./Results/{exp_args.dataset_name}/{exp_args.net}/'
#     f'TestCheckpoints_{exp_args.net}_'
#     f'{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
# )
# ────────────────────────────────────────────────────────────────────────────────────────────────


# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# #  2️⃣ ========  BLEU Detokenization (🔖 EN/FR/DE/IT/RO/NL) =======================================
# # ───────────────────────────────────────────────────────────────────────────────────────────────── 
# # ────────────────────────────────────────────────────────────────────────────────────────────────
# def detokenize_for_bleu(s: str) -> str:
#     s = s.replace("@@ ", "").replace("@@", "")  # BPE
#     s = s.replace("▁", " ")                     # SentencePiece
#     s = " ".join(s.split())                     # normalize spaces
#     s = re.sub(r"\s+([?.!,])", r"\1", s)        # punctuation
#     return s.strip()
# # ────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣ ========  BLEU Detokenization (🔖 Chinese) =================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
def detokenize_for_bleu(s: str) -> str:
    # For Chinese, don't remove SentencePiece boundaries or spaces | Only remove BPE continuation markers
    # sacreBLEU's "zh" tokenizer expects raw characters.
    return s.replace("@@ ", "").replace("@@", "").strip()
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  3️⃣ ======== Test_Checkpoint Function ===========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ test_checkpoint function
def test_checkpoint_epochs(task, model, criterion, checkpoint_dir, test_checkpoint_results_path):
    """
    Evaluate BLEU (beam=5) for all checkpoints in checkpoint_dir.
    Save results to test_checkpoint_results_path in the same format as normal test logs.
    """
# ────────────────────────────────────────────────────────────────────────────────────────────────
    global TDM_SCALE, TDM_SCALE_LOG
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Collect checkpoints sorted by epoch number in filename
    def extract_epoch_num(filename):
        match = re.search(r"epoch_(\d+)", filename)
        return int(match.group(1)) if match else float("inf")  # inf if no epoch number

    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pt")),
        key=lambda x: extract_epoch_num(os.path.basename(x))
    )

    if not ckpts:
        print(f"⚠️ No checkpoints found in {checkpoint_dir}")
        return

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # # ✅ Reset output file
    # os.makedirs(os.path.dirname(test_checkpoint_results_path), exist_ok=True)
    # with open(test_checkpoint_results_path, "w", encoding="utf-8") as f:
    #     f.write("")
    # ────────────────────────────────────────────────────────────────────────────────────────────────
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 📦 Reset output file + write model status
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 📦 Reset output file + write model status
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    #1️⃣ Create output folder if it does not exist
    os.makedirs(os.path.dirname(test_checkpoint_results_path), exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Normal run: reset file
    # ✅ Crash run : append to existing file
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣♻️ Normal run: reset file
    if not exp_args.crash_run:

        with open(test_checkpoint_results_path, "w", encoding="utf-8") as f:
            f.write("")

    # 3️⃣⏭️ Crash run : append to existing file
    else:

        with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
            f.write("\n♻️ Resuming checkpoint testing after crash\n\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣✍️  Write test result header (model information)
    with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
        f.write("")

        # ---------------------------------------------------------------------------
        # 4️⃣.1️⃣🏷️ Testing Model Status
        # ---------------------------------------------------------------------------    
    
        if exp_args.net == "TDM_Former":

            f.write(
                f"Model: TDM_Former\n"
                f"Ablation Status: "
                f"transition={exp_args.tdm_transition} | "
                f"rotation={exp_args.tdm_rotation} | "

                f"enhanced_layers=Last {exp_args.tdm_num_layers} decoder FFN layers "
                f"(L{6-exp_args.tdm_num_layers+1}-L6) | "

                f"insertion={exp_args.tdm_insertion_type}\n\n"
            )

        else:

            f.write(
                # f"Model: Baseline Fairseq Transformer\n\n"
                f"Model: Baseline Fairseq Transformer\n"
                f"Ablation Status: transition=False | rotation=False | enhanced_layers=None | insertion=None\n\n"                    
            )    
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────       
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    best_bleu = 0.0
    total_test_duration = 0.0
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    for ckpt in ckpts:

        # # ────────────────────────────────────────────────────────────────────────────────
        # # ✅ 📌📌 Load checkpoint (OLD) 🅰️🔼 
        # # ────────────────────────────────────────────────────────────────────────────────        
        # print(f"🔄 Loading checkpoint: {ckpt}")
        # state = torch.load(ckpt, map_location=DEVICE)
        # model.load_state_dict(state["model_state"], strict=False)
        # model.to(DEVICE)
        # model.eval()
        # --------------------------------------------------------------------------------------------------
        # ────────────────────────────────────────────────────────────────────────────────
        # ✅📌📌 Load checkpoint (NEW) 🅱️🔼 Restore TDM schedule for checkpoint testing using same strategy as main loop
        # ────────────────────────────────────────────────────────────────────────────────
        if exp_args.net == "Transformer":
            print(f"🔄 Loading checkpoint Baseline Transformer : {ckpt}")
            state = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state["model_state"], strict=False)
            model.to(DEVICE)
            model.eval()
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        elif exp_args.net == "TDM_Former":
            print(f"🔄 Loading checkpoint TDM_Former : {ckpt}")
            state = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state["model_state"], strict=False)            
            

            TDM_START_EPOCH = exp_args.tdm_start_epoch
            TDM_FULL_EPOCH = exp_args.tdm_full_epoch
            ckpt_epoch = state["epoch"]

            if ckpt_epoch < TDM_START_EPOCH:
                TDM_SCALE = exp_args.tdm_scale
            else:
                progress = min(
                    1.0,
                    (ckpt_epoch - TDM_START_EPOCH) / (TDM_FULL_EPOCH - TDM_START_EPOCH)
                )

                TDM_SCALE = 0.5 * (1.0 - math.cos(math.pi * progress))

            print(f"📐 Checkpoint TDM_SCALE for epoch {ckpt_epoch}: {TDM_SCALE:.3f}")

            model.to(DEVICE)
            model.eval()               
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        else:
            raise ValueError(f"Unknown model type: {exp_args.net}")
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────────────────────────────────    
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────────────────────────────────


        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Build test iterator
        test_iter = task.get_batch_iterator(
            dataset=task.dataset("test"),
            max_tokens=fairseq_args.max_tokens,   # cut in half option: fairseq_args.max_tokens // 2
            max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
            seed=seed1,
            num_shards=exp_args.num_shards,
            shard_id=exp_args.shard_id,
            num_workers=exp_args.num_workers,
            ignore_invalid_inputs=exp_args.ignore_invalid_inputs,
        ).next_epoch_itr(shuffle=False)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ References
        true_references = []
        for sample in test_iter:
            targets = sample["target"]
            for t in targets:
                t_no_pad = t[t != tgt_dict.pad()]
                ref_str = tgt_dict.string(t_no_pad, bpe_symbol="@@")
                true_references.append(detokenize_for_bleu(ref_str))
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Rebuild iterator for decoding
        test_iter = task.get_batch_iterator(
            dataset=task.dataset("test"),
            max_tokens=fairseq_args.max_tokens,    # cut in half option: fairseq_args.max_tokens // 2
            max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
            seed=seed1,
            num_shards=exp_args.num_shards,
            shard_id=exp_args.shard_id,
            num_workers=exp_args.num_workers,
            ignore_invalid_inputs=exp_args.ignore_invalid_inputs,
        ).next_epoch_itr(shuffle=False)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Build generator with beam=5
        gen_cfg = deepcopy(cfg.generation)
        gen_cfg.beam = exp_args.beam         # 🎀🔖 5
        gen_cfg.lenpen = exp_args.lenpen     # 🎀🔖 1.0 
        generator = task.build_generator([model], gen_cfg)
        # ────────────────────────────────────────────────────────────────────────────────────────────────


 

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Run evaluation
        sys_outputs, test_loss = [], 0.0
        start_time = time.time()

        with torch.no_grad():
            with tqdm(enumerate(test_iter), total=len(test_iter), desc=f"Testing {os.path.basename(ckpt)}") as progress:
                for i, sample in progress:   # 👈 keep i!
                    sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample
                    src_tokens = sample["net_input"]["src_tokens"]
                    prev_output_tokens = sample["net_input"]["prev_output_tokens"]
                    target = sample["target"]
                    src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)

                    with torch.cuda.amp.autocast():
                        logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
                        pad = tgt_dict.pad()
                        logits2d = logits.view(-1, logits.size(-1))
                        target1d = target.view(-1)
                        mask = target1d != pad
                        loss = criterion(logits2d[mask], target1d[mask]) if mask.any() else logits2d.sum() * 0.0
                    test_loss += loss.item()

                    hypos = task.inference_step(generator, [model], sample)
                    for j, hypos_j in enumerate(hypos):
                        hypo_tokens = [tok for tok in hypos_j[0]["tokens"].tolist() if tok != tgt_dict.pad()]
                        sys_str = tgt_dict.string(hypo_tokens, bpe_symbol="@@")
                        sys_outputs.append(detokenize_for_bleu(sys_str))

                    # ✅ Update progress bar postfix with running average
                    progress.set_postfix(loss=round(test_loss / (i + 1), 3))
       # ────────────────────────────────────────────────────────────────────────────────────────────────




       # ────────────────────────────────────────────────────────────────────────────────────────────────
        avg_loss = test_loss / max(1, len(test_iter))
        # bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references])                 #🔖 EN/FR/DE/IT/RO/NL
        bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references], tokenize="zh")    #🔖 Chinese

        duration = time.time() - start_time
        total_test_duration += duration
       # ────────────────────────────────────────────────────────────────────────────────────────────────


       # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Write results
        with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
            f.write(f"Checkpoint {os.path.basename(ckpt)} | Test Loss: {avg_loss:.3f} | sacreBLEU: {bleu.score:.2f} | Beam=5\n")

        print(f"✅ {os.path.basename(ckpt)} | Loss {avg_loss:.3f} | sacreBLEU {bleu.score:.2f}")


        # 🧹 add visual spacing between runs
        tqdm.write("")  
       # ────────────────────────────────────────────────────────────────────────────────────────────────

       # ────────────────────────────────────────────────────────────────────────────────────────────────
        if bleu.score > best_bleu:
            best_bleu = bleu.score

    with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n🏆 Best BLEU Score: {best_bleu:.2f}\n")
        total_mins, total_secs = divmod(total_test_duration, 60)
        f.write(f"\n🕒 Total Test Time over all checkpoints: {int(total_mins)} min {total_secs:.2f} sec\n")

    print(f"🏆 Best BLEU over checkpoints: {best_bleu:.2f}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# # ✅ Free unused GPU memory BEFORE training starts
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# 4️⃣ Call the function
test_checkpoint_epochs(task, model, criterion, checkpoint_dir, test_checkpoint_results_path)
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────




# %%
########################################################################################################################
####-------| NOTE 12.1 MODEL SUMMARY LOGGING | XXX ---------------------------------------------########################
########################################################################################################################

# ================================================================================================
# 🏷️============ 📝 Model Summary Logging =======================================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  1️⃣✅ ========  Load baseline Fairseq Transformer ==============================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
baseline_model = task.build_model(cfg.model)

# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣✅ ========  Save model summary =============================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

if exp_args.net == "TDM_Former":

    # ✅ Required only because FLOPs summary performs forward before training loop
    TDM_SCALE = exp_args.tdm_scale

    save_model_summary(
        baseline_model=baseline_model,
        model=model,
        task=task,
        train_dataset=train_dataset,
        test_dataset=test_dataset,    
        exp_args=exp_args,
        fairseq_args=fairseq_args,
        model_summary_path=model_summary_path
    )

else:

    save_model_summary(
        baseline_model=baseline_model,
        model=model,
        task=task,
        train_dataset=train_dataset,
        test_dataset=test_dataset,    
        exp_args=exp_args,
        fairseq_args=fairseq_args,
        model_summary_path=model_summary_path
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 12.2 MODEL SUMMARY LOGGING (FULL) | XXX --------------------------------------########################
########################################################################################################################

# ================================================================================================
# 🏷️============ 📝 Model full Summary Logging ==================================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  1️⃣✅ ========  Load baseline Fairseq Transformer ==============================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 
baseline_model = task.build_model(cfg.model)

# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  2️⃣✅ ========  Save full model summary ========================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

if exp_args.net == "TDM_Former":

    # ✅ Required only because FLOPs summary performs forward before training loop
    TDM_SCALE = exp_args.tdm_scale

    save_model_summary_full(
        baseline_model=baseline_model,
        model=model,
        task=task,
        train_dataset=train_dataset,
        test_dataset=test_dataset,    
        exp_args=exp_args,
        fairseq_args=fairseq_args,
        model_summary_path=model_summary_full_path
    )

else:

    save_model_summary_full(
        baseline_model=baseline_model,
        model=model,
        task=task,
        train_dataset=train_dataset,
        test_dataset=test_dataset,    
        exp_args=exp_args,
        fairseq_args=fairseq_args,
        model_summary_path=model_summary_full_path
    )
# ────────────────────────────────────────────────────────────────────────────────

# %%