Conventional image restoration models require large datasets of clean/degraded image pairs, which are expensive and often infeasible to collect in real-world settings. This project proposes a self-supervised alternative that learns entirely from unpaired, unlabeled degraded images by combining a Vision Transformer encoder with a diffusion-based denoiser trained via masked patch noise estimation.

Architecture:
Degraded Image x₀
         │
    ┌────▼────────────┐
    │   ViT Encoder   │   patch tokenization + positional embeddings
    └────┬────────────┘
         │
    ┌────▼──────────────────────┐
    │  Transformer Denoiser     │   multi-head self-attention
    │  conditioned on (x_t, t)  │   cross-attention over context patches
    └────┬──────────────────────┘
         │   predicted noise ε_θ(x_t, t)
    ┌────▼───────────────────┐
    │  DDPM / DDIM Reverse   │   iterative denoising
    └────┬───────────────────┘
         │
   Restored Image x̂₀

Features:
1. Self-supervised training with no paired ground-truth images
2. Transformer-based denoiser with multi-head self-attention and learned timestep conditioning
3. Supports Gaussian denoising, blind deconvolution, super-resolution (×2, ×4), and inpainting
4. Linear, cosine, and sigmoid beta noise schedules
5. DDIM sampling with 50 steps (~20× faster than DDPM-1000)
6. FP16 mixed precision, gradient checkpointing, and EMA weight averaging
7. Modular design — swap backbone, schedule, or sampler independently

   
