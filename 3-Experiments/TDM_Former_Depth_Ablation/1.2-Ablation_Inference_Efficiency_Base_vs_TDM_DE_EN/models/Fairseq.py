



# %% 

#####------------------------- NOTE MY MODEL IWSLT14 DE-EN NOTE -----------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################### NOVEL LIGHTWEIGHT MODEL ##############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE MY MODEL IWSLT14 DE-EN NOTE ------------------------------------------------------#####






########################################################################################################################
########################################################################################################################
####-------| NOTE 🛠️ IMPORT AND SETUP ⚙️ | XXX ---------------------------------------------------#####################
########################################################################################################################
########################################################################################################################


# 📄 Fairseq.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣.1️⃣📜 ============ Import Standard libraries & torch libraries ==============================
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 === Enable flexible CUDA memory allocation to reduce fragmentation ==========================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 === Core Libraries ==========================================================================
import sys
import torch
import numpy as np
import math
import random
print("Python Version:", sys.version.split()[0])     # Should be 3.10.16
print("PyTorch Version:", torch.__version__)         # Should be 2.1.0
print("CUDA Available:", torch.cuda.is_available())  # Should be True
print("CUDA Version:", torch.version.cuda)           # Should be 11.8

import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2

from ptflops import get_model_complexity_info
from calflops import calculate_flops
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣.2️⃣📜 ============ Import Fairseq & Supporting Imports ======================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
import importlib.metadata as importlib_metadata
import fairseq  # type: ignore
from fairseq.data import Dictionary, data_utils, iterators          # type: ignore
from fairseq.tasks.translation import TranslationTask               # type: ignore
from fairseq import options                                         # type: ignore
from fairseq.dataclass.utils import convert_namespace_to_omegaconf  # type: ignore                                             
from fairseq import utils                                           # type: ignore 
import hydra                                                        # type: ignore  
# ────────────────────────────────────────────────────────────────────────────────────────────────
from fairseq.models import FairseqEncoderDecoderModel                              # type: ignore
from fairseq.models.fairseq_encoder import FairseqEncoder                          # type: ignore
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder   # type: ignore
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



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣📦 ============ Define directory ============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
PROJECT_PATH = r"c:\Users\emeka\Research\ModelCUDA\Transformers\Transformer_Baselines_DE_EN"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣.1️⃣📜 ============  Imput parser safe for afno because of --fno-bias, etc  ==================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_IWSLT14De_En.py
from parser_IWSLT14De_En import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
# args, unknown = parser.parse_known_args()

# ✅ IMPORTANT: Do NOT read Jupyter / VSCode kernel arguments
# This prevents the "--f" ambiguity issue
exp_args = parser.parse_args(args=[])

num_aug_splits = exp_args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import Fairseq parser/config
from fairseq_config_IWSLT14De_En import get_fairseq_parser
# ✅ Build Fairseq configuration
fairseq_args, cfg = get_fairseq_parser(exp_args)


num_aug_splits = exp_args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")

print(f"✅ Encoder Embed Dim: {fairseq_args.encoder_embed_dim}")
print(f"✅ Decoder Layers: {fairseq_args.decoder_layers}")

# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣.2️⃣📜 ============ Seeding for reproducibility ==============================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
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
########################################################################################################################
####-------| NOTE 📐 FAIRSEQ TRANSFORMER SPECIFICATION AND ARCHITECTURE 🔑 | XXX -----------------#####################
########################################################################################################################
########################################################################################################################


# ================================================================================================
# 🏷️1. =============== Define Path and Initialization 🔥🔮 =====================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Model Specification & Architecture ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────

modelspec_path = {
    "log_modelspec_history":
        f'./Results_Architecture_Summary/{exp_args.dataset_name}/{exp_args.net}/'
        f'{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_'
        f'{exp_args.mode_name}_model_complexity_architecture.txt'
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧾 === Initialize histories and logs ===========================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
log_modelspec_history = []
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# TDM_SCALE = exp_args.tdm_scale

# ================================================================================================
# 📊🏷️2. ============  Model Complexity Check ===================================================
# ================================================================================================
# ==================🟢🔴🔍📉📊🏷️3🧩🏷️📌🔥📊📐🏷️3.🔑🔑⚖️🔬⚙️🔧🧬🚫🔒➕==================
# 🎀🟦✅🟩🟨🟧🟥📉🎛️⭐✔🔑⏪⏭️📦♻️✔️🎯🚀❌⚠❤️💛🔵⚪🌊⚖️🧩🔖🧠🥇🥈🥉👍🚦🔍========
# ================================================================================================

# =============================================================================
# 🧠1️⃣ Build Fairseq task
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔧 Ensure correct project root
PROJECT_PATH2 = r"c:\Users\emeka\Research\ModelCUDA\Transformers\Transformer_Baselines_DE_EN"

os.chdir(PROJECT_PATH2)

print("✅ Current working directory:")
print(os.getcwd())
# ─────────────────────────────────────────────────────────────────────────────────────────────────

task = TranslationTask.setup_task(cfg.task)

# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ================================================================================================
# 2️⃣ 🧠 Build Fairseq Transformer Baseline 📐🔑🏷️
# ================================================================================================
# ================================================================================================
# 2️⃣ 🧠 Build Fairseq Transformer Baseline 📐🔑🏷️
# ================================================================================================

if exp_args.net == "Transformer":

    print(f"♻️♻️ Loaded Model: Transformer baseline")
    print(f"♻️♻️ Loaded Model: Transformer baseline")

    # =============================================================================
    # 📌📌 Load baseline Fairseq Transformer
    # =============================================================================
    model = task.build_model(cfg.model)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    print("\nMODEL INIT CHECK")
    print(model.decoder.layers[0].fc1.weight[0, :10])
    print()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    print(f"✅ Loaded Model: Transformer")
    print(f"✅ Loaded Model: {model.__class__.__name__}")

else:
    raise ValueError(f"Unknown model type for baseline summary: {exp_args.net}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
log_modelspec_history.append("================================================================================================")
log_modelspec_history.append(f"📊🏷️1. ============= {exp_args.net} Complexity Check ===========================")
log_modelspec_history.append("================================================================================================")


# =============================================================================
# 🧠 Model summary
# =============================================================================
model_total_params = sum(p.numel() for p in model.parameters())
model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 📊 MACs / FLOPs / Params
# =============================================================================
model_flops, model_macs, model_params = calculate_flops(

    model=model,

    kwargs={

        "src_tokens": torch.randint(
            0,
            len(task.source_dictionary),
            (1, exp_args.max_source_positions)
        ),

        "src_lengths": torch.tensor(
            [exp_args.max_source_positions]
        ),

        "prev_output_tokens": torch.randint(
            0,
            len(task.target_dictionary),
            (1, exp_args.max_target_positions)
        )
    },

    print_results=False
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# =============================================================================
# 📜 Logging
# =============================================================================
log_modelspec_history.append(f"✅ {exp_args.net} Total Parameters: {model_total_params:,}")
# log_modelspec_history.append(f"✅ {exp_args.net} Trainable Parameters: {model_trainable_params:,}")
log_modelspec_history.append(f"⚙️ {exp_args.net} MACs: {model_macs}")
log_modelspec_history.append(f"🔥 {exp_args.net} FLOPs: {model_flops}")
log_modelspec_history.append(f"📦 {exp_args.net} Params: {model_params}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# =============================================================================
# ⚖️ Transformer/PTN configuration
# =============================================================================
log_modelspec_history.append(
    f"⚖️ model={exp_args.net} "
    f"| encoder_layers={fairseq_args.encoder_layers} "
    f"| decoder_layers={fairseq_args.decoder_layers} "
    f"| encoder_embed_dim={fairseq_args.encoder_embed_dim} "
    f"| decoder_embed_dim={fairseq_args.decoder_embed_dim} "
    f"| max_target_positions={exp_args.max_target_positions} "
    f"| dropout={fairseq_args.dropout}"
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# =============================================================================
# 🔧 Experiment configuration
# =============================================================================
log_modelspec_history.append(
    f"🔧 activation={exp_args.act_name} "
    f"| optimizer={exp_args.optimizer1} "
    f"| lr={exp_args.lr}"
)


# print("\n".join(log_modelspec_history))

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ================================================================================================
# 📊🏷️3. ============ Implementation Graph ======================================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

log_modelspec_history.append(f"\n==============================================================================================")
log_modelspec_history.append(f"📐🏷️2. ============= {exp_args.net} Architecture ===============================")
log_modelspec_history.append(f"================================================================================================")



# =============================================================================
# 🧠 Full model architecture
# =============================================================================
log_modelspec_history.append(str(model))
print("\n".join(log_modelspec_history))



# ================================================================================================
# 🏷️4. ============ Architectural Specification ==================================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

log_modelspec_history.append(f"\n==============================================================================================")
log_modelspec_history.append(f"📐🏷️3. ============= {exp_args.net} Architectural Specification  ===============")
log_modelspec_history.append(f"================================================================================================")



# ================================================================================================
# 🔒 ============== Save Logs & Training Results (once per epoch) 📦 ============================
# ================================================================================================
# ================================================================================================
####------------------------------------------------------------------------------------------####

# ✅ === Save Train Results ===
os.makedirs(os.path.dirname(modelspec_path["log_modelspec_history"]), exist_ok=True)

with open(modelspec_path["log_modelspec_history"], "w") as f:
    f.write("\n".join(log_modelspec_history))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# ────────────────────────────────────────────────────────────────────────────────────────────────




# %%


