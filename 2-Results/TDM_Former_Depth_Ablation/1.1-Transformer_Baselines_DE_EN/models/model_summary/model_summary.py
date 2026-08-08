



# %% 

#####------------------------- NOTE MODEL SUMMARY IWSLT14 DE-EN NOTE ------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################### IWSLT14 DE-EN ########################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE MODEL SUMMARY IWSLT14 DE-EN NOTE -------------------------------------------------#####



# 📄 model_summary.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣📜 ============ Import Standard libraries & torch libraries  ===================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
import torch
import sys
import os
import math
from ptflops import get_model_complexity_info
from calflops import calculate_flops
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣📦 ============ Define directory ============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📄 model_summary.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ================================================================================================
# 3️⃣⚛️ MODEL SUMMARY LOGGING
# ================================================================================================
# ================================================================================================
# 3️⃣⚛️ MODEL SUMMARY LOGGING
# ================================================================================================



# ================================================================================================
# 🏷️ ============ 📝 Model Summary Logging ======================================================
# ================================================================================================
def save_model_summary(
    baseline_model,    
    model,
    task,
    train_dataset,
    test_dataset,    
    exp_args,
    fairseq_args,
    model_summary_path
):




    # =============================================================================
    # 🧠 Current model summary
    # =============================================================================
    model_total_params = sum(p.numel() for p in model.parameters())

    model_trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # =============================================================================
    # 📊 Current model MACs / FLOPs / Params
    # =============================================================================
    if exp_args.decimal_places == "two":

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

    elif exp_args.decimal_places == "four":

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

            print_results=False,
            output_as_string=False
        )

    else:
        raise ValueError(f"❌ Unknown decimal_places option: {exp_args.decimal_places}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # =============================================================================
    # 🧠 Fairseq baseline Transformer summary
    # =============================================================================

    baseline_total_params = sum(
        p.numel()
        for p in baseline_model.parameters()
    )

    baseline_trainable_params = sum(
        p.numel()
        for p in baseline_model.parameters()
        if p.requires_grad
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # =============================================================================
    # 📊 Fairseq baseline MACs / FLOPs / Params
    # =============================================================================
    if exp_args.decimal_places == "two":

        baseline_flops, baseline_macs, baseline_params = calculate_flops(

            model=baseline_model,

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

    elif exp_args.decimal_places == "four":

        baseline_flops, baseline_macs, baseline_params = calculate_flops(

            model=baseline_model,

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

            print_results=False,
            output_as_string=False
        )

    else:
        raise ValueError(f"❌ Unknown decimal_places option: {exp_args.decimal_places}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────────
    # ✅ Save model summary
    # ────────────────────────────────────────────────────────────────────────────
    model_summary = []
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # =============================================================================
    # Dataset Information
    # =============================================================================
    model_summary.append("------------------------------------------")
    model_summary.append("Dataset Information")
    model_summary.append("------------------------------------------")

    model_summary.append(
        f"🌍 dataset_name= "
        f"{exp_args.dataset_name}"
    )

    model_summary.append(
        f"🌍 translation_direction= "
        f"{fairseq_args.source_lang} → {fairseq_args.target_lang}"
    )

    model_summary.append(
        f"📊 train_samples= "
        f"{len(train_dataset):,}"
    )

    model_summary.append(
        f"📊 test_samples= "
        f"{len(test_dataset):,}"
    )

    model_summary.append("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # =============================================================================
    # Current model
    # =============================================================================
    model_summary.append("------------------------------------------")
    model_summary.append("Current Model")
    model_summary.append("------------------------------------------")

    model_summary.append(f"Model name: {exp_args.net}")

    model_summary.append("")

    model_summary.append(
        f"✅ {exp_args.net} Total Parameters: "
        f"{model_total_params:,}"
    )

    model_summary.append(
        f"✅ {exp_args.net} Trainable Parameters: "
        f"{model_trainable_params:,}"
    )

    model_summary.append("")

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if exp_args.decimal_places == "two":

        model_summary.append(
        f"📦 Params: {baseline_params} M | "
        f"⚙️ MACs: {baseline_macs} GMACs | "
        f"🔥 FLOPs: {baseline_flops} GFLOPS"
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    elif exp_args.decimal_places == "four":

        model_summary.append(
        f"📦 Params: {model_params / 1e6:.4f} M | "
        f"⚙️ MACs: {model_macs / 1e9:.4f} GMACs | "
        f"🔥 FLOPs: {model_flops / 1e9:.4f} GFLOPS"
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(f"❌ Unknown decimal_places option: {exp_args.decimal_places}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    model_summary.append("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # =============================================================================
    # RTM Parameters
    # =============================================================================
    if exp_args.net == "RTM_Former":

        model_summary.append("------------------------------------------")
        model_summary.append("RTM Parameters")
        model_summary.append("------------------------------------------")

        model_summary.append(
        f"🔧 rtm_alpha= "
        f"{exp_args.rtm_alpha} | "
        f"🔧 rtm_transition_amplification= "
        f"{exp_args.rtm_transition_amplification} | "
        f"🔧 rtm_num_layers= "
        f"{exp_args.rtm_num_layers} | "
        f"⏪⏭️ rtm_insertion_type= "
        f"{exp_args.rtm_insertion_type}"
        )

        model_summary.append("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # =============================================================================
    # Fairseq Baseline Transformer Parameters
    # =============================================================================
    model_summary.append("------------------------------------------")
    model_summary.append("Fairseq Baseline Transformer")
    model_summary.append("------------------------------------------")

    model_summary.append(
        f"🏗️ model_class= "
        f"{baseline_model.__class__.__name__}"
    )

    model_summary.append("")

    model_summary.append(
        f"✅ Fairseq Total Parameters: "
        f"{baseline_total_params:,}"
    )

    model_summary.append(
        f"✅ Fairseq Trainable Parameters: "
        f"{baseline_trainable_params:,}"
    )

    model_summary.append("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if exp_args.decimal_places == "two":

        model_summary.append(
        f"📦 Params: {baseline_params} M | "
        f"⚙️ MACs: {baseline_macs} GMACs | "
        f"🔥 FLOPs: {baseline_flops} GFLOPS"
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    elif exp_args.decimal_places == "four":

        model_summary.append(
        f"📦 Params: {baseline_params / 1e6:.4f} M | "
        f"⚙️ MACs: {baseline_macs / 1e9:.4f} GMACs | "
        f"🔥 FLOPs: {baseline_flops / 1e9:.4f} GFLOPS"
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(f"❌ Unknown decimal_places option: {exp_args.decimal_places}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    model_summary.append("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    model_summary.append(
        f"⚖️ encoder_layers= "
        f"{fairseq_args.encoder_layers} | "
        f"decoder_layers= "
        f"{fairseq_args.decoder_layers} | "
        f"encoder_embed_dim= "
        f"{fairseq_args.encoder_embed_dim} | "
        f"decoder_embed_dim= "
        f"{fairseq_args.decoder_embed_dim} | "
        f"max_target_positions= "
        f"{exp_args.max_target_positions} | "
        f"dropout= "
        f"{fairseq_args.dropout}"
    )

    model_summary.append("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # =============================================================================
    # Optimizer + Activation
    # =============================================================================
    model_summary.append("------------------------------------------")
    model_summary.append("Optimiser & Activation")
    model_summary.append("------------------------------------------")

    model_summary.append(
        f"🔧 activation= "
        # f"{fairseq_args.activation_fn} | "
        f"{exp_args.act_name} | "
        f"optimizer= "
        f"{exp_args.optimizer1} | "
        f"lr= "
        f"{exp_args.lr} | "
        f"smoothing= "
        f"{exp_args.smoothing}"
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # =============================================================================
    # Create directory
    # =============================================================================
    os.makedirs(
        os.path.dirname(model_summary_path),
        exist_ok=True
    )

    # =============================================================================
    # Save summary
    # =============================================================================
    with open(model_summary_path, "w", encoding="utf-8") as f:

        f.write("\n".join(model_summary))

    print(f"📝 Model summary saved to: {model_summary_path}")

# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# ────────────────────────────────────────────────────────────────────────────────────────────────




# %%
