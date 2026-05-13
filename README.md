# Morphology-Aware Deep Learning for Real-Time Uterine Tumour Detection

This repository contains the **M-ACAM** implementation used in the associated research: a morphology-aware pipeline for **uterine tumour detection** (and related segmentation / attention components) on ultrasound-style imagery, implemented in **PyTorch**.

The `m_acam` package holds the model (detection-oriented backbone with segmentation / attention parts), dataset loading, losses, training and evaluation entry points, and auxiliary scripts used to produce tables and figures for the manuscript.

## Results in the paper vs. your own runs

The **numerical results reported in the paper correspond to our best run** under our fixed experimental setup (data split, seeds where applicable, hyperparameters, hardware, and software versions at the time of the experiments).

**You should not expect to reproduce those numbers exactly** on another machine. Differences in GPU / CPU, driver and CUDA versions, PyTorch and library builds, operating system, dataloader workers, floating-point non-determinism, and small changes in data or preprocessing can all shift metrics. Treat the paper values as **reference outcomes** for our reported configuration, not as a strict checksum for every environment.

## License and citation

If you use this code or build on these ideas, please cite the corresponding paper when it is available, and respect the license file if one is added to the repository.
