# TDM-Former

**TDM-Former: Transition Dynamics Modulation for Adaptive Token Dynamics in Transformer Neural Machine Translation**

---

## Overview

TDM-Former extends the decoder feed-forward network (FFN) ResBlock of the standard Transformer by introducing Transition Dynamics Modulation (TDM).

TDM captures the relationship between consecutive activated FFN representations and uses this relationship to modulate the current activated FFN representation before the second FFN linear transformation.

---

## Baseline FFN ResBlock

The conventional Transformer FFN ResBlock contains a position-wise FFN with two fully connected linear sublayers and an activation function. Each FFN output is computed from the corresponding activated FFN representation without directly using the preceding activated FFN representation.

<p align="center">
  <img src="1-Architecture/ffn_resblock.svg" width="450"/>
</p>

<p align="center">
  <b>Figure: Baseline FFN ResBlock</b>
</p>

---

## TDM-Enhanced FFN ResBlock

The proposed TDM-enhanced FFN ResBlock incorporates TDM between the FFN activation and the second linear transformation. TDM computes a transition signal between consecutive activated FFN representations and uses this signal to modulate the current activated FFN representation.

<p align="center">
  <img src="1-Architecture/ffn_tdm_resblock.svg" width="450"/>
</p>

<p align="center">
  <b>Figure: TDM-enhanced FFN ResBlock</b>
</p>

---

## Repository Structure

- `1-Architecture`: Architecture diagrams for the baseline and TDM-enhanced FFN ResBlocks.
- `2-Results`: Translation results, ablation results, efficiency measurements, and plots.
- `3-Experiments`: Training, evaluation, and model implementation files for the evaluated IWSLT translation tasks.

---

## 📄 Architecture Figures (PDF)

- [Baseline FFN ResBlock](1-Architecture/ffn_resblock.pdf)
- [TDM-Enhanced FFN ResBlock](1-Architecture/ffn_tdm_resblock.pdf)









---
## 📊 Translation Results

| Task | Transformer Baseline | TDM-Former | BLEU Gain | Relative Gain | TDM Epoch | Baseline Epoch |
|:---|---:|---:|---:|---:|---:|---:|
| EN&nbsp;→&nbsp;DE | 25.34 | **25.76** | +0.42 | +1.66% | 48 | 48 |
| DE&nbsp;→&nbsp;EN | 30.81 | **31.51** | +0.70 | +2.27% | 47 | 42 |
| EN&nbsp;→&nbsp;RO | 26.31 | **26.63** | +0.32 | +1.22% | 47 | 49 |
| RO&nbsp;→&nbsp;EN | 32.88 | **33.60** | +0.72 | +2.19% | 49 | 49 |
| EN&nbsp;→&nbsp;IT | 28.63 | **28.78** | +0.15 | +0.52% | 49 | 49 |
| IT&nbsp;→&nbsp;EN | 31.87 | **32.63** | +0.76 | +2.38% | 46 | 46 |
| EN&nbsp;→&nbsp;ZH | 20.28 | **20.60** | +0.32 | +1.58% | 49 | 49 |
| ZH&nbsp;→&nbsp;EN | 34.35 | **34.96** | +0.61 | +1.78% | 46 | 46 |






---

## 🧪 Ablation Results
**IWSLT14 DE&nbsp;→&nbsp;EN TDM insertion-depth ablation**

| Model Variant | BLEU | Gain | Relative Gain | Best Epoch |
|:---|---:|---:|---:|---:|
| Transformer&nbsp;Baseline | 30.81 | +0.00 | +0.00% | 42 |
| TDM-Former&nbsp;(L6) | 31.16 | +0.35 | +1.14% | 47 |
| TDM-Former&nbsp;(L5–L6) | 31.37 | +0.56 | +1.82% | 47 |
| **TDM-Former&nbsp;(L4–L6)** | **31.51** | **+0.70** | **+2.27%** | **47** |
| TDM-Former&nbsp;(L3–L6) | 31.54 | +0.73 | +2.37% | 47 |





---

**IWSLT14 DE&nbsp;→&nbsp;EN TDM insertion-depth ablation**

| Model Variant | Runs | BLEU Last | BLEU Best | Inference Time (s) | Time Δ | Throughput (words/s) | Throughput Δ | Memory (GB) | Memory Δ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer&nbsp;Baseline | 10 | 30.48&nbsp;±&nbsp;0.00 | 30.81 | 6.03&nbsp;±&nbsp;0.33 | +0.00% | 5204.05&nbsp;±&nbsp;290.83 | +0.00% | 1.74&nbsp;±&nbsp;0.00 | +0.00% |
| TDM-Former&nbsp;(L6) | 10 | 31.16&nbsp;±&nbsp;0.00 | 31.16 | 6.40&nbsp;±&nbsp;0.37 | +6.05% | 4985.42&nbsp;±&nbsp;289.68 | −4.20% | 1.74&nbsp;±&nbsp;0.00 | +0.00% |
| TDM-Former&nbsp;(L5–L6) | 10 | 31.16&nbsp;±&nbsp;0.00 | 31.37 | 6.54&nbsp;±&nbsp;0.49 | +8.37% | 4870.09&nbsp;±&nbsp;349.02 | −6.42% | 1.74&nbsp;±&nbsp;0.00 | +0.00% |
| **TDM-Former&nbsp;(L4–L6)** | **10** | **31.34&nbsp;±&nbsp;0.00** | **31.51** | **6.80&nbsp;±&nbsp;0.45** | **+12.70%** | **4660.17&nbsp;±&nbsp;288.56** | **−10.45%** | **1.74&nbsp;±&nbsp;0.00** | **+0.00%** |
| TDM-Former&nbsp;(L3–L6) | 10 | 31.34&nbsp;±&nbsp;0.00 | 31.54 | 7.11&nbsp;±&nbsp;0.57 | +17.84% | 4485.32&nbsp;±&nbsp;328.22 | −13.81% | 1.74&nbsp;±&nbsp;0.00 | +0.00% |

---
