# [EnsCLR](https://kenncoder7.github.io/2025/01/30/EnsCLR/)
This is an official PyTorch implementation of "EnsCLR: Unsupervised skeleton-based action recognition via ensemble contrastive learning of representation" in CVIU.

How to learning the meaningful features of unlabeled data in unsupervised manner is a key question in AI models. Thus, we proposed a ensemble contrastive learning framework for unsupervised representation learning, named EnsCLR. Finally, EnsCLR was applied to the unsupervised skeleton-based action representation learning task for practice implementation.

![The framework of EnsCLR.](https://github.com/KennCoder7/EnsCLR/blob/main/weights/framework.png)

## Data Preparation
The data preparation is on the basis of [AimCLR](https://github.com/Levigty/AimCLR?tab=readme-ov-file).

## Paper Info
- **Title**: EnsCLR: Unsupervised skeleton-based action recognition via ensemble contrastive learning of representation
- **Author(s)**: K Wang, J Cao, B Cao, B Liu
- **Conference/Journal Name**: Computer Vision and Image Understanding
- **Date**: 2024
- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1077314224001577)
- **GitHub**: [EnsCLR](https://github.com/KennCoder7/EnsCLR)
  
## Abstract
Skeleton-based action recognition is a key research area in video understanding, beneficial from its compact and efficient motion information. To relieve from the burden of expensive and laborious data annotation, unsupervised approaches, particularly contrastive learning, have been widely employed to extract action representations from unlabeled data. In this paper, we propose an Ensemble framework for Contrastive Learning of Representation (EnsCLR) to preform unsupervised skeleton-based action recognition. Concretely, Queue Extension method is devised to generate discriminative representation by aggregating the ensemble information from multiple pipelines. Furtherly, Ensemble Nearest Neighbors Mining (ENNM) method is utilized to excavate the most similar samples from the unlabeled data as positive samples, which alleviates the false-negative samples problem caused by the disregard of category label. The experiments with extensive evaluation protocols show that EnsCLR outperforms previous state-of-the-art methods on NTU60, NTU120, and PKU-MMD datasets.

## Citation
If you find this idea helpful, please consider citing:
```
@article{wang2024ensclr,  
  title={EnsCLR: Unsupervised skeleton-based action recognition via ensemble contrastive learning of representation},  
  author={Wang, Kun and Cao, Jiuxin and Cao, Biwei and Liu, Bo},  
  journal={Computer Vision and Image Understanding},  
  volume={247},  
  pages={104076},  
  year={2024},  
  publisher={Elsevier}  
}
```

The framework of our code is extended from the following repositories [CrosSCLR](https://github.com/LinguoLi/CrosSCLR) and [AimCLR](https://github.com/Levigty/AimCLR?tab=readme-ov-file). We sincerely thank the authors for releasing the codes.
