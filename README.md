# Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction
## Basic info:
This is the release code for :
[Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction](https://arxiv.org/pdf/2205.14526) 
which is accepted by SIGKDD 2022!

Recommended ref:
```
Dongjie Wang, Yanjie Fu, Kunpeng Liu, Xiaolin Li, Yan Solihin. Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction. The 28-th ACM SIGKDD Conference, 2022
```

Recommended Bib:
```
@InProceedings{wang2022group,
  title={Group-wise Reinforcement Feature Generation for Optimal and Explainable Representation Space Reconstruction},
  author={Wang, Dongjie and Fu, Yanjie and Liu, Kunpeng and Li, Xiaolin and Solihin, Yan},
  journal={The 28-th ACM SIGKDD Conference},
  year={2022}
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
