# Overview
This repository contains official implementation for our paper titled "Improving Normative Modeling for Multi-modal Neuroimaging Data using mixture-of-product-of-experts variational autoencoders", accepted in IEEE International Symposium in Biomedical Imaging (IEEE ISBI 2024). [[ArXiV](https://arxiv.org/pdf/2312.00992.pdf)]

<img align="center" width="65%" height="80%" src="Plots/workflow.png"> 

Figure 1:  Proposed MoPoE normative modelling framework

## Abstract

Normative models in neuroimaging learn the brain patterns of healthy population distribution and estimate how disease subjects like Alzheimer's Disease (AD) deviate from the norm. Existing variational autoencoder (VAE)-based normative models using multimodal neuroimaging data aggregate information from multiple modalities by estimating product or averaging of unimodal latent posteriors. This can often lead to uninformative joint latent distributions which affects the estimation of subject-level deviations. In this work, we addressed the prior limitations by adopting the Mixture-of-Product-of-Experts (MoPoE) technique which allows better modelling of the joint latent posterior. Our model labelled subjects as outliers by calculating deviations from the multimodal latent space. Further, we identified which latent dimensions and brain regions were associated with abnormal deviations due to AD pathology.with patient cognition and result in higher number of brain regions with statistically significant deviations compared to the unimodal baseline model.

# Implementation details

## Environment & Packages

We recommend an environment with python >= 3.7 and pytorch >= 1.10.2, and then install the following dependencies:
```
pip install -r requirements.txt
```
All models were implemented using the multi-view-AE package developed by Aguila, Ana Lawry, et al [Multi-view-AE: A Python package for multi-view autoencoder models](https://joss.theoj.org/papers/10.21105/joss.05093).

Cortical and subcortical brain atlases were visualized using the ggseg package. The original R implementation can be found [here](https://github.com/ggseg/ggseg). A more recent Python implementation can also be found [here](https://github.com/ggseg/python-ggseg).

## Datasets & Feature extraction

### Datasets

We used the ADNI dataset in our study. ADNI data are available through an access procedure described at [http://adni.loni.usc.edu/data-samples/ access-data/](http://adni.loni.usc.edu/data-samples/ access-data/).

### Feature extraction

We used preprocessed brain regions’ volumes from the T1-weighted MRI images. These brain region volumes were preprocessed through the FreeSurfer software (version 5.1). The cortical surface of each hemisphere was parcellated according to the Desikan–Killiany atlas and anatomical volumetric measures were obtained via a whole-brain segmentation procedure. The final data included cortical regions(32 per hemisphere) and 24 subcortical regions (12 per hemisphere). 


## Performance Evaluation
<img align="center" width="65%" height="100%" src="Plots/clinical_validation.png"> 

<img align="center" width="65%" height="100%" src="Plots/interpretability.png"> 


## Acknowledgement

The preparation of this report was supported by the Centene Corporation contract (P19-00559) for the Washington University-Centene ARCH Personalized Medicine Initiative and the National Institutes of Health (NIH) (R01-AG067103). Computations were performed using the facilities of the Washington University Research Computing and Informatics Facility, which were partially funded by NIH grants S10OD025200, 1S10RR022984-01A1 and 1S10OD018091-
01. Additional support is provided The McDonnell Center for Systems Neuroscience
  
## Citation
If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@article{kumar2023improving,
  title={Improving Normative Modeling for Multi-modal Neuroimaging Data using mixture-of-product-of-experts variational autoencoders},
  author={Kumar, Sayantan and Payne, Philip and Sotiras, Aristeidis},
  journal={arXiv preprint arXiv:2312.00992},
  year={2023}
}

