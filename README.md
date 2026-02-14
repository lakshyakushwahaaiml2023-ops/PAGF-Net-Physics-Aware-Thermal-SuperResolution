# PAGF-Net-Physics-Aware-Thermal-SuperResolution
Physics-Aware Optical-Guided Deep Learning Network for High-Resolution Thermal Infrared Image Reconstruction using Energy-Constrained Super-Resolution.

PAGF-Net v2: Physics-Aware Guided Fusion for Satellite Thermal SR
PAGF-Net v2 (Physics-Aware Guided Fusion Network) is a deep learning framework designed to super-resolve low-resolution (LR) satellite thermal imagery by leveraging high-resolution (HR) optical guidance. By utilizing Spatial Feature Transform (SFT) layers and a Physics-Aware Loss function, the model preserves structural boundaries and maintains energy conservation—critical for quantitative thermal analysis.

#Core Features
SFT Modulation: Unlike simple concatenation, our SFT layers learn to affine-transform thermal features based on optical structural priors.

Physics-Aware Loss: A multi-component loss function integrating:

Fidelity (L1): Pixel-wise reconstruction.

Structural (Sobel): Edge preservation.

Energy (AvgPool): Downsampling consistency with LR observations.

Diffusion (Laplacian): Preservation of heat-diffusion patterns.

Deep Supervision: Dual-stage reconstruction (Coarse-to-Fine) to ensure stable gradient flow.

Inference Refinement: Includes Iterative Back-Projection (IBP) and 8-point Geometric Ensembling (TTA) for maximum performance.

#Model Architecture
The network follows a progressive refinement strategy:

Shallow Extraction: Parallel branches for Thermal and Optical inputs.

SFT Backbone: 8 Residual SFT Blocks split into two stages.

Intermediate Supervision: A "Coarse" output is generated mid-network to guide early layers.

Final Refinement: Residual learning to recover high-frequency thermal details.

##Experiment journey:

# Phase 1: Optical-Guided Residual Network (Baseline)

###  Status: Completed 
**Role:** Initial Proof of Concept  
**Architecture:** ResNet + Channel Attention (SE-Block)  
**Key Innovation:** Introduction of "Physics-Aware Loss"

---

## Experiment Overview

This was our first attempt to solve the Thermal Super-Resolution problem. Instead of using a standard black-box upscale (like Bicubic or SRCNN), we hypothesized that **Optical Images (RGB)** could provide the "structural skeleton" for the thermal data.

We also introduced a custom **Physics Loss** to ensure the total thermal energy remained constant during upscaling.

### Model Architecture
* **Backbone:** Residual Blocks with Channel Attention (RCAN-style).
* **Fusion:** Early concatenation of Optical and Thermal features.
* **Upsampling:** Residual Learning (predicting the difference, not the raw pixel values).

### Loss Function
We moved beyond standard MSE loss by defining a compound loss:
$$L_{total} = L_{reconstruction} + 0.5 \times L_{energy}$$

* **$L_{reconstruction}$:** L1 Loss (Pixel accuracy).
* **$L_{energy}$:** Forces the downsampled super-resolution output to match the original low-resolution input (Conservation of Energy).

---

## Results (Phase 1)

| Metric | Value |
| :--- | :--- |
| **Training Epochs** | 20 |
| **PSNR** | ~34.5 dB (Est.) |
| **SSIM** | ~0.91 |
| **RMSE** | 0.034 |

###  Why we moved to Phase 2 (Limitations)
1.  **Texture Bleeding:** The optical textures sometimes "overpowered" the thermal data, creating fake edges where temperature changes didn't actually exist.
2.  **Limited Receptive Field:** The standard Residual Blocks struggled to understand large-scale context (e.g., big lakes vs. small ponds).
3.  **Aliasing:** The simple upsampling caused minor jagged edges.

> *This experiment proved the concept of "Optical Guidance" worked, paving the way for the advanced Attention Mechanisms used in the final PAGF-Net v2.*

---

##  How to Run This Baseline

# Train the baseline
python train.py

# Evaluate on test set
python eval_unseen.py

# Phase 2: Spatial Feature Transform (SFT) Network

###  Status: Completed (Significant Improvement)
**Role:** Advanced Intermediate Model  
**Architecture:** SFT-ResNet + Deep Supervision  
**Key Innovation:** Spatial Feature Transform (SFT) Layers

---

##  Experiment Overview

After observing "Texture Bleeding" in Phase 1, we hypothesized that simple concatenation was too aggressive. The model was forcing optical textures onto thermal data even when they didn't match.

In **Phase 2**, we implemented **Spatial Feature Transform (SFT)**.
Instead of *adding* the optical image, the network generates **Scale ($\gamma$)** and **Shift ($\beta$)** maps from the optical data to modulate the thermal features.

$$\text{Feature}_{out} = \text{Feature}_{in} \cdot (\gamma + 1) + \beta$$

This allows the model to "turn off" optical guidance in areas where it isn't helpful (e.g., a cold cloud covering a hot fire).

###  Key Architectural Changes
1.  **SFT Layers:** Replaced standard addition with affine transformations conditioned on the Optical image.
2.  **Deep Supervision:** We added an auxiliary loss halfway through the network (`mid_output`). This forces the early layers to learn useful features immediately, preventing vanishing gradients.
3.  **Laplacian Loss:** We added a loss term that focuses specifically on high-frequency edges (using Laplacian filters) to sharpen the blurry thermal borders.



---

##  Results (Phase 2)

| Metric | Phase 1 (Baseline) | Phase 2 (SFT) | Improvement |
| :--- | :--- | :--- | :--- |
| **PSNR** | 34.50 dB | **36.10 dB** | +1.6 dB |
| **SSIM** | 0.910 | **0.935** | +2.5% |
| **Visual Quality** | Blurry edges | **Sharp edges** | Significant |

###  Why we moved to Final Version (Limitations)
1.  **Computational Cost:** SFT layers are computationally expensive. The training time doubled compared to Phase 1.
2.  **Grid Artifacts:** The Deep Supervision occasionally caused "checkerboard" artifacts because the intermediate output forced the network to upscale too early.
3.  **Lack of Global Context:** While SFT is great for local details, the model still struggled with large-scale consistency (e.g., consistent temperature across a large lake).

> *This phase proved that **Modulation > Concatenation**. We carried this lesson into the final PAGF-Net v2 but optimized it for speed and global context.*

---

##  How to Run This Phase

# Train the SFT model
python train_advanced.py
# Test the SFT model
python test_advanced.py



#Phase 3: PAGF-Net v2 (The Final Model)
 Status: Final Production Version
Role: Optimized High-Performance Network

Architecture: Multi-Scale SFT-ResNet + Physics-Anchored Supervision

Key Innovation: Sequential Progressive Modulation & Physics Anchoring

 Experiment Overview
PAGF-Net v2 is the culmination of our research. In this version, we addressed the "Grid Artifacts" and "Computational Inefficiency" seen in Phase 2. We moved from a simple auxiliary loss to a Physics-Anchored Deep Supervision strategy.

Instead of just predicting a coarse image in the middle, we use the Bicubic Input as a Physics Anchor. This ensures that even at the intermediate stage, the features are anchored to the original low-resolution thermal measurements, preventing the "hallucination" of thermal data that doesn't exist.

Key Architectural Refinements
Sequential Stage Split: We split the 8 SFT blocks into two distinct stages (Stage 1: Coarse, Stage 2: Fine). This allows the network to focus on structural alignment first and texture refinement second.

Physics-Anchored Loss: The PhysicsAwareLoss now monitors both the mid_output and the final_output, ensuring energy conservation is maintained at every depth of the network.

Leaky Mapping: Replaced standard ReLU in the SFT mapping layers with LeakyReLU (0.1) to prevent "dying neurons" in the guidance branch, leading to better edge recovery.


###Results of all Models are shown seperately in the Research Models folder
