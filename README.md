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
