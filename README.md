# CEID Computer Vision & Graphics — Exercises Repository

This repository contains the assignments for the **Computer Vision & Graphics** lab (CEID, Department of Computer Engineering & Informatics, University of Patras). Each folder corresponds to one exercise, including problem statements, code and example outputs.

## Table of Contents

1. [Exercise 1: Gaussian & Laplacian Pyramids](https://github.com/GrigorisTzortzakis/Computer-Vision/blob/main/Exercise%201/Theory/CV_PYRAMIDS.pdf)  
2. [Exercise 2: Basic Geometric Transformations](https://github.com/GrigorisTzortzakis/Computer-Vision/blob/main/Exercise%202/Theory/CV_TRANSFORMATIONS.pdf)  
3. [Exercise 3: Scale-Invariant Feature Transform (SIFT)](https://github.com/GrigorisTzortzakis/Computer-Vision/blob/main/Exercise%203/Theory/CV_3-SIFT.pdf)  
4. [Exercise 4: Image Alignment & ECC/LK Methods](https://github.com/GrigorisTzortzakis/Computer-Vision/blob/main/Exercise%204/Theory/project_CV.pdf)  
5. [Exercise 5: PCA, Autoencoders & Variational Autoencoders](https://github.com/GrigorisTzortzakis/Computer-Vision/blob/main/Exercise%205/Theory/CV_5_AUTOENCODERS.pdf)  

---

## Exercise 1: Gaussian & Laplacian Pyramids

**Topics Covered:**  
- Multi‑scale image decomposition via **Gaussian** and **Laplacian pyramids**  
- Building and blending image pyramids for denoising, feature extraction, compression and mosaicking  
- Implementing pyramids from first principles (filter design, convolution, decimation/interpolation) and verifying reconstruction  
- Hands‑on MATLAB toolbox usage (`gen_Pyr`, `pyr_Reduce`, `pyr_Expand`, `pyrBlend`, `pyr_Reconstruct`)  
- Brief overview of **Spatial Pyramid Pooling (SPP)** in deep CNNs and **Pyramid Pooling** for segmentation  

**Objective:**  
- Understand theory and practice of image pyramids  
- Implement Gaussian and Laplacian pyramid construction and blending  
- Apply to seamless image stitching and explore SPP layer concepts  

---

## Exercise 2: Basic Geometric Transformations

**Topics Covered:**  
- Familiarization with MATLAB/OpenCV functions: `imread`, `imwarp`, `affine2d`, `projective2d`, `imref2d`, `implay`  
- Image scaling pyramid: compose an image of multiple scaled versions of itself  
- Periodic shearing animation of a sample image (`pudding.png`) and save as video  
- Wind‑mill compositing: apply rotation, scaling, translation with masks for natural blending (`windmill_mask.png`, `windmill.png`, `windmill_back.jpeg`)  
- Compare interpolation methods (`nearest`, `linear`, `cubic`) and document quality differences  
- Ball animation over beach background: design custom trajectory and degeneration at the horizon  

**Objective:**  
- Gain hands‑on experience with affine and projective warps in MATLAB/OpenCV  
- Create a variety of animated sequences demonstrating geometric transformations  

---

## Exercise 3: Scale-Invariant Feature Transform (SIFT)

**Topics Covered:**  
- **Theory** of SIFT in four stages:  
  1. **Scale‑space extrema** detection via Difference of Gaussians  
  2. **Keypoint localization**: selecting stable extrema, rejecting low‑contrast or edge responses  
  3. **Orientation assignment**: computing gradient histograms around each keypoint  
  4. **Descriptor formation**: 16×16 neighborhood histograms (4×4×8‑bin) for rotation invariance  
- Detailed equations for Gaussian convolution, DoG, gradient magnitude/orientation, and histogram voting  

**Objective:**  
- Implement or use an existing SIFT pipeline  
- Visualize detected keypoints and descriptors  
- Understand the mathematical foundations of scale and rotation invariance  

---

## Exercise 4: Image Alignment & ECC / Lucas–Kanade Methods

**Topics Covered:**  
- MATLAB scripts for image alignment:  
  - `ecc_lk_alignment.m`: compares **ECC** (Enhanced Correlation Coefficient) vs. **Lucas–Kanade** methods across multiple pyramid levels  
  - `spatial_interp.m`: inverse warping + interpolation (`interp2`) to sample warped image  
  - `image_jacobian.m` + `warp_jacobian.m`: computing image and warp Jacobians for parameter updates  
  - `param_update.m`: updating transformation parameters (translation, Euclidean, affine, homography)  
- Experiments aligning video frames (high vs. low resolution; pure translation vs. combined warps)  
- Analysis of **convergence speed**, **PSNR**, and effect of resolution on alignment quality  

**Objective:**  
- Dive deep into two popular iterative registration algorithms  
- Measure performance and robustness under different distortions  

---

## Exercise 5: PCA, Autoencoders & Variational Autoencoders

**Topics Covered:**  
1. **Principal Component Analysis (PCA)**  
   - Derivations via variance maximization and minimum projection error  
   - Eigenvalue decomposition vs. SVD formulations  
   - Hands‑on on MNIST: mean digit, covariance, first 8 PCs, reconstructions for L = 1, 8, 16, 64, 256; error histograms  
   - Kernel PCA: effect of centered vs. uncentered mappings, Gaussian kernel experiments  
2. **Autoencoders (AE)**  
   - Undercomplete (bottleneck) networks, linear vs. nonlinear, MSE vs. binary‐crossentropy loss  
   - Connections between linear AE and PCA  
3. **Variational Autoencoders (VAE)**  
   - Probabilistic latent variables, encoder to parameterize posterior (`μ`, `σ`), KL divergence regularization  
   - Reparameterization trick and decoder network  

**Objective:**  
- Compare PCA, (V)AE for dimensionality reduction and generative modeling  
- Implement networks, visualize reconstructions and latent traversals  

---

