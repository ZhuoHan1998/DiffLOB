# DiffLOB: Diffusion Models for Counterfactual Generation in Limit Order Books

This repository contains the official implementation of **DiffLOB**, a score-based diffusion framework for **controllable and counterfactual generation of limit order book (LOB) trajectories** under future market regime interventions.

DiffLOB explicitly conditions the generative process on **future market regimes**—including **trend, volatility, liquidity, and order-flow imbalance**—and enables direct counterfactual queries of the form:

> *“If the future market regime were different, how would the LOB evolve?”*

---

## 🔍 Overview

Existing generative models for limit order books primarily focus on reproducing realistic market dynamics, but remain fundamentally **passive**: they model what typically happens, rather than what would happen under hypothetical future conditions.

DiffLOB addresses this limitation by:
- Formulating LOB generation as a **conditional score-based diffusion problem**
- Explicitly introducing **future market regimes** as control variables
- Enabling **counterfactual interventions** without relying on agent-based interaction
- Demonstrating the usefulness of counterfactual trajectories for downstream tasks

---

## ✨ Key Features

- **Score-based diffusion model (VP-SDE)** for LOB trajectories
- Explicit conditioning on **future market regimes**
- **ControlNet-style control module** for regime-aware intervention
- **Two-stage training strategy** for stable controllable generation
- **Classifier-free guidance** for flexible conditional sampling
- **Ancestral sampling** of the reverse-time SDE
- Joint modeling of **price and volume dynamics**
- Extensive evaluation on realism, counterfactual validity, and downstream usefulness


<p align="center">
  <img src="pics/DiffLOB_Architecture.png" width="700">
</p>

<p align="center">
  <img src="pics/price_trends_AMZN.png" width="700">
</p>


<p align="center">
  <img src="pics/volume_heatmaps_AMZN.png" width="700">
</p>
