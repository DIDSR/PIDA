# PIDA
## Physics-Informed Data Augmentation to simulate low dose CT scans: Application to Lung Nodule Detection
Our proposed Physics-Informed Data Augmentation (PIDA) method lever-
ages the mAs and Noise Power Spectrum (NPS) profiles of various CT reconstruction
kernels to simulate the effects of various dose exposures. In this approach, the NPS
of a higher dose CT scan is used to generate correlated noise, which is then stochas-
tically inserted into the training data. This simulates the noise characteristics of the
lower dose exposure and enhances variability within the training set. To demonstrate
PIDA’s applicability in improving the generalizability of CNNs, we applied PIDA in
training a neural network designed to reduce false positives in a lung nodule detection
algorithm. We evaluated the impact of the noise insertion training method by assessing
lung nodule detection performance on low-dose CT scans.
