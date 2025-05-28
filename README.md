# Physics-Informed Data Augmentation (PIDA) to simulate low dose CT scans: Application to Lung Nodule Detection
This repository contains the code for **Physics-Informed Data Augmentation to simulate low dose CT scans: Application to Lung Nodule Detection**.
Our proposed **Physics-Informed Data Augmentation (PIDA)** method leverages the mAs and Noise Power Spectrum (NPS) profiles of various CT reconstruction
kernels to simulate the effects of various dose exposures. In this approach, the NPS of a higher dose CT scan is used to generate correlated noise, which is then stochastically inserted into the training data to simulate the noise characteristics of the
lower dose exposure. We applied PIDA in training a neural network designed to reduce false positives in a lung nodule detection algorithm. We evaluated the impact of the noise insertion training method by assessing
lung nodule detection performance on low-dose CT scans.

## Visualization of Nodule after PIDA
<p align="center">
  <img src="figures/cropped_nodules_2.png" alt="Example Image" width="600"/>
</p>
<p align="center"><em>Figure 1: Visualization of nodule candidate after our proposed physics-informed noise insertion technique. The top row shows a) a sample CT scan from LIDC-IDRI dataset, acquired at 400 mAs and reconstructed using a standard filter, and b) a cropped ROI, and the bottom row shows c) the cropped ROI without noise, d) with white-Gaussian noise, and e) with noise insertion based on our PIDA method. Images are shown with the same window level of -1000 to 400 HU.</em></p>


##  Repository Structure
```bash
PIDA/
├── figures/                  # Visualizations and final result plots
├── annotations/              # Required for preprocessing nodules
├── src/                      # Core Python scripts for train and test
│   ├── baseline_train.py     # Train CNN only on High Dose (HD) CT images of LIDC dataset
│   ├── baseline_test.py      # Test trained baseline on HD, Low-Dose (LD), and Standard-Dose (SD) CT images
│   └── PIDA_train.py         # Train baseline with PIDA on High Dose (HD) CT images
│   ├── PIDA_test.py          # Test on HD, LD, and SD CT images
│   ├── WGDA_train.py         # Train baseline with White-Gaussian (WG) Noise on High Dose (HD) CT images
│   └── WGDA_test.py          # Test on HD, LD, and SD CT images
│   ├── baseline_with_GA.py   # Train baseline with Geometric Augmentation (i.e., random rotation, scaling, transpose & flip) on 9 Folds of LUNA16 challenge, and tested on 10th Fold.
│   ├── PIDA_with_GA.py       # Train baseline with PIDA and Geometric Augmentation (i.e., random rotation, scaling, transpose & flip) on 9 Folds of LUNA16 challenge, and tested on 10th Fold.
│   └── WGDA_with_GA.py       # Train baseline with WG noise and Geometric Augmentation (i.e., random rotation, scaling, transpose & flip) on 9 Folds of LUNA16 challenge, and tested on 10th Fold.
├── model_checkpoints/        # model checkpoints for each of the network mentioned as *_train.py file in src directory
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation (this file)
cd .

## 🧪 How to Run

1. Install required Python packages:
   ```bash
   pip install -r requirements.txt

