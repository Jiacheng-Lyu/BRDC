# multi-omics BRDC
This repository includes the code used in BRDC multi-comic study.

**Proteogenomic Landscape of Human Breast Ductal Carcinoma Progression**

Ganfei Xu, Juan Yu, Jiacheng Lyu, Mengna Zhan, Jie Xu, Minjing Huang, Rui Zhao, Yan Li, Jiajun Zhu, Subei Tan, Peng Ran, Zhenghua Su, Jun Chang, Jianyuan Zhao, Hongwei Zhang, Chen Xu, Xinhua Liu, Yingyong Hou, and Chen Ding

## Code overview
> The below figure numbers were corresponded to the paper version.

### 1. Figure1.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code of the proteogenomic analysis among all stages in the BRDC

Output figures and tables:  
* Figure 1C, 1D, 1E

### 2. Figure2.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code of the proteogenomic analysis between Lesion and DCIS stages.

Output figures and tables:  
* Figure 2A, 2B, 2D, 2E, 2F, 2G, 2H, 2I, 2J, and 2L
* Supplementary Table 3E, 3I

### 3. Figure3.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code of the proteogenomic analysis between Lesion and DCIS_Pure.

Output figures and tables:  
* Figure 3A, 3B, 3C, 3D, 3E, 3F, 3G, 3H, and 3I

### 4. Figure4.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code of the proteogenomic analysis between DCIS_Pure and DCIS_Pro.

Output figures and tables:  
* Figure 4A, 4B, 4C, 4D, 4E, 4F, 4G, 4H, 4I, 4J, and 4L
* Supplementary Table 5A, 

### 5. Figure5.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code of the proteogenomic analysis including 4 clinical subtypes (LuminalA, LuminalB, Her2-enriched, and TNBC) of IDC, and DCIS_Pro-IDC pair-wised progression.

Output figures and tables:  
* Figure 5B, and 5C
* Supplementary Table 6A, 6B, 6C, and 6D

### 6. Figure6.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code of the proteogenomic analysis between IDC and M-Distant.

Output figures and tables:  
* Figure 6A, 6B, 6C, 6D, 6E, 6F, 6G, 6I, and 6J
* Supplementary Table 7B

## Environment requirement
The following package/library versions were used in this study:
* python (version 3.9.15)
* pandas (version 1.5.2)
* numpy (version 1.24.1)
* scipy (version 1.9.3)
* statsmodels (version 0.13.5)
* lifelines (version 0.27.4)
* matplotlib (version 3.6.2)
* seaborn (version 0.11.2)
* statannotations (version 0.5.0)

## Folders Structure
The files are organised into four folders:
* *document*: which contains all the genomic, transcriptomic, proteomics, phosphoproteomic and clinical patient informations required to perform the analyses described in the paper. The data files is currently deposited to the zenodo repository and can be available on Supplementary_Tables link.
* *code*: contains the python code in the ipython notebook to reproduce all analyses and generate the the figures in this study.
* *documents*: contains the related Supplementary Table produced by the code.
* *figure*: contains the related plots produced by the code.
