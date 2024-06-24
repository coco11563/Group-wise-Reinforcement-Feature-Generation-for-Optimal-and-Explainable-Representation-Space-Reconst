# Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction
## Basic info:
This is the release code for :
[Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction](https://arxiv.org/pdf/2205.14526) 
which is accepted by SIGKDD 2022!

See also: Our code repo ([code](https://github.com/coco11563/Traceable_Automatic_Feature_Transformation_via_Cascading_Actor-Critic_Agents)), a Actor-Critic version feature transformation

Recommended ref:
```
Dongjie Wang, Yanjie Fu, Kunpeng Liu, Xiaolin Li, Yan Solihin. Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction. The 28-th ACM SIGKDD Conference, 2022
```

Recommended Bib:
```
KDD version bib:
@InProceedings{wang2022group,
  title={Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction},
  author={Wang, Dongjie and Fu, Yanjie and Liu, Kunpeng and Li, Xiaolin and Solihin, Yan},
  journal={The 28-th ACM SIGKDD Conference},
  year={2022}
}
KDD extended paper:
@article{xiao2024traceable,
  title={Traceable group-wise self-optimizing feature transformation learning: A dual optimization perspective},
  author={Xiao, Meng and Wang, Dongjie and Wu, Min and Liu, Kunpeng and Xiong, Hui and Zhou, Yuanchun and Fu, Yanjie},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={18},
  number={4},
  pages={1--22},
  year={2024},
  publisher={ACM New York, NY}
}
Following Actor-Critic RL Framework paper: 
@inproceedings{xiao2023traceable,
  title={Traceable automatic feature transformation via cascading actor-critic agents},
  author={Xiao, Meng and Wang, Dongjie and Wu, Min and Qiao, Ziyue and Wang, Pengfei and Liu, Kunpeng and Zhou, Yuanchun and Fu, Yanjie},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={775--783},
  year={2023},
  organization={SIAM}
}
```
***
## Paper Abstract

Representation (feature) space is an environment where data points are vectorized,  distances are computed, patterns are characterized, and geometric structures are embedded. Extracting a good representation space  is critical to address the curse of dimensionality, improve model generalization, overcome data sparsity, and increase the availability of classic models. 
Existing literature, such as feature engineering and representation learning, is limited in achieving full automation (e.g., over heavy reliance on intensive labor and empirical experiences), explainable explicitness (e.g., traceable reconstruction process and explainable new features), and flexible optimal (e.g., optimal feature space reconstruction is not embedded into downstream tasks).  
Can we simultaneously address the automation, explicitness, and optimal challenges in representation space reconstruction for a machine learning task?
To answer this question, we propose a  group-wise reinforcement generation perspective. 
We reformulate representation space reconstruction into an interactive process of nested feature generation and selection, where feature generation is to generate new meaningful and explicit features, and feature selection is to eliminate redundant features to control feature sizes. 
We develop a cascading reinforcement learning method that leverages three cascading Markov Decision Processes to learn optimal generation policies to automate the selection of features and operations and the feature crossing.
We design a group-wise generation strategy to cross a feature group, an operation, and another feature group to generate new features and find the strategy that can enhance exploration efficiency and augment reward signals of cascading agents.
Finally, we present extensive experiments to demonstrate the effectiveness, efficiency, traceability, and explicitness of our system.
***


## How to run:
### step 1: download the code and dataset:
```
git clone git@github.com:coco11563/Group-wise-Reinforcement-Feature-Generation-for-Optimal-and-Explainable-Representation-Space-Reconst.git
```
then:
```
follow the instruction in readme.md in `/data/processed/data_info.md` to get the dataset
```

### step 2: run the code with main script:`main.py`

```
xxx/python3 main.py --name DATASETNAME --episodes SEARCH_EP_NUM --steps SEARCH_STEP_NUM...
```

please check each configuration in `initial.py`

### step 3: enjoy the generated dataset:

the generated feature will in ./tmp/NAME_SAMPLINE_METHOD_/xxx.csv
