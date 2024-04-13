# Overview
This repository contains official implementation for our paper titled "Improving Normative Modeling for Multi-modal Neuroimaging Data using mixture-of-product-of-experts variational autoencoders", accepted in IEEE International Symposium in Biomedical Imaging (IEEE ISBI 2024). [[ArXiV](https://arxiv.org/pdf/2312.00992.pdf)]


## Abstract

Normative models in neuroimaging learn the brain patterns of healthy population distribution and estimate how disease subjects like Alzheimer's Disease (AD) deviate from the norm. Existing variational autoencoder (VAE)-based normative models using multimodal neuroimaging data aggregate information from multiple modalities by estimating product or averaging of unimodal latent posteriors. This can often lead to uninformative joint latent distributions which affects the estimation of subject-level deviations. In this work, we addressed the prior limitations by adopting the Mixture-of-Product-of-Experts (MoPoE) technique which allows better modelling of the joint latent posterior. Our model labelled subjects as outliers by calculating deviations from the multimodal latent space. Further, we identified which latent dimensions and brain regions were associated with abnormal deviations due to AD pathology.with patient cognition and result in higher number of brain regions with statistically significant deviations compared to the unimodal baseline model.

# Implementation details

## Environment & Packages

We recommend an environment with python >= 3.7 and pytorch >= 1.10.2, and then install the following dependencies:
```
pip install -r requirements.txt
```
Cortical and subcortical brain atlases were visualized using the ggseg package. The original R implementation can be found [here](https://github.com/ggseg/ggseg). A more recent Python implementation can also be found [here](https://github.com/ggseg/python-ggseg).
