# Diffusion Trajectory Analysis

This project studies the **geometry of inference trajectories** in generative models focusing on four models:
- **DDPM** 
- **DDIM** 
- **Latent Diffusion** (Stable Diffusion)
- **Flow Matching** (Stable Diffusion 3, Rectified Flow Matching)


Rather than evaluating image quality, this work analyzes **how samples move through space during inference**


---

## Installation

This project is **inference-only** and does not require training.

All dependencies are listed in `requirements.txt`.

Using Conda:
```bash
conda create -n diffusion-analysis python=3.10
conda activate diffusion-analysis
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

---

## Project Structure

```
project/
├── DDPM/
├── DDIM/
├── Stable_diffusion/
├── flow_matching/
├── data/
├── notebooks/
├── extract_steps.py
├── requirements.txt
└── README.md
```

### DDPM 

Implements DDPM inference on CIFAR-10.

**Files**
- `model_loader.py`: loads pretrained DDPM model and scheduler.
- `inference.py`: does the reverse denoising process
- `run_diffusion.py`: entry point for running DDPM and its inference.

---


### DDIM

Deterministic variant of diffusion.

**Files**
- `model_loader.py`: loads the DDIM model and scheduler
- `inference.py`: does the reverse denoising process
- `run_ddim.py` : entry point for running DDIM and its inference

---

### Stable Diffusion — Latent Diffusion Model

**Files**
- `model_loader.py`
- `inference.py`
- `run_diffusion.py`

The file structure is the same as DDIM and DDPM

---

### Flow Matching (Stable Diffusion 3)



**Files**
- `model_loader.py`
- `inference.py`
- `run_fm.py`

The file Structure is the same
---

### extract_steps.py

This file is a utility module for saving intermediate inference steps to disk. Although not central to the analysis notebook as the notebook extracts the intermediates directly from the inference codes, it provides reproducibility and debugging support.
---
### data/

Optional directory for storing intermediate steps by running `extract_steps.py`. 

---

### notebooks

- `Analysis.ipynb` is a notebook that has all the experimental analysis done. It imports the models from the files mentioned earlier and does inference on the model for further analysis.

In the notebook we do the analysis on how the intermediate steps during generation behave, comparing :


- **DDPM** 
- **DDIM** 
- **Latent Diffusion** 
- **Flow Matching (SD3)** 

#### 1. Latent Trajectory Extraction

For each model, we extract the sequence of latent states:

$$
\{x_T, x_{T-1}, \dots, x_0\}
$$

where $$ \( x_t \in \mathbb{R}^{C \times H \times W} \) $$ represents either:
- a pixel-space tensor (DDPM / DDIM), or
- a latent-space tensor (Latent Diffusion / Flow Matching).

This trajectory is the basis for the next analyses.

---

#### 2. Trajectory Length

We measure the **total path length** traveled by the model during inference:

$$
L = \sum_{t=1}^{T} \| x_{t-1} - x_t \|_2
$$


---

#### 3. Per-Step Displacement

We analyze how much the latent changes **at each inference step**:

$$
\Delta_t = \| x_{t-1} - x_t \|_2
$$

This hows the displacement of each step as part of the generation process.


---

#### 4. Directional Alignment

This experiment studies **directional consistency** of inference steps.

We define:

**Global direction**

$$
d_{\text{global}} = \frac{x_0 - x_T}{\|x_0 - x_T\|_2}
$$

**Local step direction**

$$
d_t = \frac{x_{t-1} - x_t}{\|x_{t-1} - x_t\|_2}
$$

**Directional alignment**

$$
\text{align}(t) = \langle d_t,\ d_{\text{global}} \rangle
$$

This measures how well each step aligns with the overall denoising direction.

---

#### 5. Trajectory Curvature

We measure the **second-order finite difference** of the trajectory:

$$
\kappa_t = \| x_{t-1} - 2x_t + x_{t+1} \|_2
$$

This captures how sharply the trajectory bends as the curvature would describe how smooth the path is.



---

#### 6. Normalized Trajectory Length

To compare trajectories across different latent dimensionalities, we normalize path length by the number of latent dimensions:

$$
L_{\text{norm}} = \frac{1}{CHW} \sum_{t=1}^{T} \| x_{t-1} - x_t \|_2
$$

This enables fair comparison between:
- pixel-space diffusion,
- latent-space diffusion.

---

#### 7. Individual Pixel / Latent Coordinate Trajectories

We track individual coordinates over time:

$$
x_t(c, h, w)
$$

This shows how each pixel changes over time

---

### 8. RGB Trajectories of a Fixed Pixel

For pixel-space models, we track the **RGB values of a fixed spatial location** across inference:

$$
(x_t^R,\ x_t^G,\ x_t^B)
$$

This shows how color channels evolve jointly during denoising and highlights differences across models during generation.

---

### 9. Interactive Visualization of Inference Steps

An interactive slider-based visualization allows manual inspection of how the generated image evolves step-by-step showing the image and the sample during each step.

This qualitative tool complements quantitative metrics and helps build intuition about:
- early vs late denoising stages,
- differences in convergence behavior.

---
