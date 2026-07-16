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
  <img src="1-Architecture/ffn_resblock.svg" width="350"/>
</p>

<p align="center">
  <b>Figure: Baseline FFN ResBlock</b>
</p>

---

## TDM-Enhanced FFN ResBlock

The proposed TDM-enhanced FFN ResBlock incorporates TDM between the FFN activation and the second linear transformation. TDM computes a transition signal between consecutive activated FFN representations and uses this signal to modulate the current activated FFN representation.

<p align="center">
  <img src="1-Architecture/ffn_tdm_resblock.svg" width="350"/>
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
