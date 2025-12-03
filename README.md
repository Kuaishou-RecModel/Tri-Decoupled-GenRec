
# A Survey of Generative Recommendation from a Tri-Decoupled Perspective: Tokenization,Architecture, and Optimization
![Paper](https://img.shields.io/badge/Paper-Your%20Paper%20Link-red)
![GitHub stars](https://img.shields.io/github/stars/Kuaishou-RecModel/Tri-Decoupled-GenRec?style=social)
![Last commit](https://img.shields.io/github/last-commit/Kuaishou-RecModel/Tri-Decoupled-GenRec)


## Overview

This is the official repository of the paper ["*A Survey of Generative Recommendation from a Tri-Decoupled Perspective: Tokenization,Architecture, and Optimization*"](https://arxiv.org/abs/2406.01171)

This paper provides an in-depth survey on the latest advancements in generative recommendation systems, focusing on key components such as tokenization, architecture design, and optimization strategies. It explores the paradigm shift from traditional discriminative models to generative models and their potential to revolutionize recommendation systems across various industries.

We continuously maintain this paper collection to foster future endeavors.

## Citation
If you find this survey useful, please cite the following paper:

```bibtex
@misc{your2025generative,
  title={Generative Recommendation Systems: A Comprehensive Survey},
  author={Your Name and Collaborators},
  year={2025},
  eprint={your-arxiv-id},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
}
```


## Table of Contents
- [Introduction](#introduction)
- [Survey Scope](#survey-scope)
- [Key Components](#key-components)
  - [Tokenization](#tokenization)
  - [Architecture Design](#architecture-design)
  - [Optimization Strategies](#optimization-strategies)
- [Applications](#applications)
  - [Industrial Use Cases](#industrial-use-cases)
  - [Challenges and Future Directions](#challenges-and-future-directions)
- [Acknowledgement](#acknowledgement)
- [How to Contribute](#how-to-contribute)
<!-- - [Citation](#citation) -->
<!-- - [Authors](#authors) -->

## Introduction
Generative recommendation systems represent a fundamental shift in how user preferences and item recommendations are modeled. Unlike traditional systems that focus on scoring predefined candidate items, generative models aim to directly generate relevant item identifiers, mitigating the cascading errors common in multi-stage discriminative systems. This survey delves into the evolution, technical foundations, and practical applications of generative recommendation systems.

<!-- - Number of publications on generative recommendation indexed in OpenAlex
![Statistics Image](figures/statistics_v2.png) -->

- Generative VS Discriminative
![Compared Image](figures/ParadigmCompare_v3.png)


## Survey Scope
This survey covers a wide range of topics relevant to generative recommendation systems, including:
- The transition from discriminative to generative models.
- Detailed analysis of tokenization, including sparse ID, text-based, and semantic ID approaches.
- Architectural frameworks like encoder-decoder, decoder-only, and diffusion-based models.
- Optimization techniques, such as reinforcement learning for preference alignment.

## Key Components

### Tokenization
Generative recommendation systems benefit significantly from the tokenization of items and user interactions. We explore three main tokenization strategies:
- **Sparse ID-based Tokenization:** Traditional approach but limited in semantics.

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Code                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *GRU4Rec* |      [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)      | ICLR'16 |                        [![Github](https://img.shields.io/github/stars/hidasib/GRU4Rec.svg?style=social&label=Github)](https://github.com/hidasib/GRU4Rec)
| *SASRec* |      [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)      | ICDM'18 |                        [![Github](https://img.shields.io/github/stars/kang205/SASRec.svg?style=social&label=Github)](https://github.com/kang205/SASRec)
| *BERT4Rec* |      [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)      | CIKM'19 |                        [![Github](https://img.shields.io/github/stars/FeiSun/BERT4Rec.svg?style=social&label=Github)](https://github.com/FeiSun/BERT4Rec)
| *HSTU* |      [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)      | ICML'24 |                        [![Github](https://img.shields.io/github/stars/meta-recsys/generative-recommenders.svg?style=social&label=Github)](https://github.com/meta-recsys/generative-recommenders)                        |
| *LUM* |      [Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model](https://arxiv.org/abs/2502.08309)      | arXiv'25 |                        /
| *MTGR* |      [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654)      | CIKM '25 |                        /                        |
| *PinRec* |      [PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems](https://arxiv.org/abs/2504.10507)      | arXiv '25 |                        /                        |
| *DFGR* |      [Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation](https://www.arxiv.org/abs/2505.16752)      | arXiv '25 |                        /                        |
| *GenRank* |      [Towards Large-scale Generative Ranking](https://arxiv.org/abs/2505.04180)      | arXiv '25 |                        /                        |

- **Text-based Tokenization:** Leverages the power of natural language models, improving semantic understanding.

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Code                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *P5* |      [Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/abs/2203.13366)      | RecSys'22 |                        [![Github](https://img.shields.io/github/stars/jeykigung/P5.svg?style=social&label=Github)](https://github.com/jeykigung/P5)
| *M6-Rec* |      [M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems](https://arxiv.org/abs/2205.08084)      | KDD'22 |                        /
| *GPT4Rec* |      [GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://arxiv.org/abs/2304.03879)      | arXiv'23 |                        /
| *RecSysLLM* |      [Leveraging Large Language Models for Pre-trained Recommender Systems](https://arxiv.org/abs/2308.10837)      | arXiv'23 |                        /
| *TALLRec* |      [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation](https://arxiv.org/abs/2305.00447)      | RecSys'23 |                        [![Github](https://img.shields.io/github/stars/SAI990323/TALLRec.svg?style=social&label=Github)](https://github.com/SAI990323/TALLRec)
| *GenRec* |      [GenRec: Large Language Model for Generative Recommendation](https://arxiv.org/abs/2307.00457)      | ECIR'24 |                        /
| *LLaRA* |      [LLaRA: Large Language-Recommendation Assistant](https://arxiv.org/abs/2312.02445)      | SIGIR'24 |                        [![Github](https://img.shields.io/github/stars/ljy0ustc/LLaRA.svg?style=social&label=Github)](https://github.com/ljy0ustc/LLaRA)
| *HLLM* |      [HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling](https://arxiv.org/abs/2409.12740)      | arXiv'24 |                        [![Github](https://img.shields.io/github/stars/bytedance/HLLM.svg?style=social&label=Github)](https://github.com/bytedance/HLLM)
| *S-DPO* |      [On Softmax Direct Preference Optimization for Recommendation](https://arxiv.org/abs/2406.09215)      | NeurIPS'24 |                 [![Github](https://img.shields.io/github/stars/chenyuxin1999/S-DPO.svg?style=social&label=Github)](https://github.com/chenyuxin1999/S-DPO) 
| *InteraRec* |      [InteraRec: Screenshot Based Recommendations Using Multimodal Large Language Models](https://arxiv.org/abs/2403.00822)      | arXiv'24 |                /
| *LLMTreeRec* |      [LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/abs/2404.00702)      | COLING'25 |                        [![Github](https://img.shields.io/github/stars/Applied-Machine-Learning-Lab/LLMTreeRec.svg?style=social&label=Github)](https://github.com/Applied-Machine-Learning-Lab/LLMTreeRec)
| *BIGRec* |      [A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems](https://arxiv.org/abs/2308.08434)      | TORS'25 |                        [![Github](https://img.shields.io/github/stars/SAI990323/BIGRec.svg?style=social&label=Github)](https://github.com/SAI990323/BIGRec)
| *Rec-R1* |      [Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning](https://arxiv.org/abs/2503.24289)      | TMLR'25 |                        [![Github](https://img.shields.io/github/stars/linjc16/Rec-R1.svg?style=social&label=Github)](https://github.com/linjc16/Rec-R1)
| *RecFound* |      [Generative Representational Learning of Foundation Models for Recommendation](https://arxiv.org/abs/2506.11999)      | arXiv'25 |                        [![Github](https://img.shields.io/github/stars/JunkFood436/RecFound.svg?style=social&label=Github)](https://github.com/JunkFood436/RecFound)



- **Semantic ID-based Tokenization:** Combines the best of both worlds with efficient, semantic-rich representations.

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Code                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *TIGER* |      [Recommender systems with generative retrieval](https://arxiv.org/abs/2305.05065)      | NeurIPS'23 |                        /
| *LC-Rec* |      [Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/abs/2311.09049)      | ICDE'24 |                        [![Github](https://img.shields.io/github/stars/RUCAIBox/LC-Rec.svg?style=social&label=Github)](https://github.com/RUCAIBox/LC-Rec)
| *LMIndexer* |      [Language Models As Semantic Indexers](https://arxiv.org/abs/2310.07815)      | ICML'24 |                        [![Github](https://img.shields.io/github/stars/PeterGriffinJin/LMIndexer.svg?style=social&label=Github)](https://github.com/PeterGriffinJin/LMIndexer)
| *TokenRec* |      [TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation](https://arxiv.org/abs/2406.10450)      | TKDE'24 |                 [![Github](https://img.shields.io/github/stars/Quhaoh233/TokenRec.svg?style=social&label=Github)](https://github.com/Quhaoh233/TokenRec)     
| *LETTER* |      [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314)      | CIKM'24 |                        [![Github](https://img.shields.io/github/stars/HonghuiBao2000/LETTER.svg?style=social&label=Github)](https://github.com/HonghuiBao2000/LETTER)
| *EAGER* |      [EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration](https://arxiv.org/abs/2406.14017)      | KDD'24 |                        [![Github](https://img.shields.io/github/stars/yewzz/EAGER.svg?style=social&label=Github)](https://github.com/yewzz/EAGER)
| *ETEGRec* |      [Generative Recommender with End-to-End Learnable Item Tokenization](https://arxiv.org/abs/2409.05546)      | SIGIR'25 |                        [![Github](https://img.shields.io/github/stars/RUCAIBox/ETEGRec.svg?style=social&label=Github)](https://github.com/RUCAIBox/ETEGRec)
| *OneRec* |      [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965)      | arXiv'25 |                       /
| *BBQRec* |      [BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation](https://arxiv.org/abs/2504.06636)      | arXiv'25 |                       /
| *OneLoc* |      [OneLoc: Geo-Aware Generative Recommender Systems for Local Life Service](https://arxiv.org/abs/2508.14646)      | arXiv'25 |                       /
| *OneSearch* |      [OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search](https://arxiv.org/abs/2509.03236)      | arXiv'25 |                       /
| *RPG* |      [Generating Long Semantic IDs in Parallel for Recommendation](https://arxiv.org/abs/2506.05781)      | KDD'25 |                   [![Github](https://img.shields.io/github/stars/facebookresearch/RPG_KDD2025.svg?style=social&label=Github)](https://github.com/facebookresearch/RPG_KDD2025)   
| *URI* |      [Making Transformer Decoders Better Differentiable Indexers](https://openreview.net/pdf?id=bePaRx0otZ)      | ICLR'25 |                /
| *EAGER-LLM* |      [EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration](https://arxiv.org/abs/2502.14735)      | WWW'25 |                /
| *UNGER* |      [UNGER: Generative Recommendation with A Unified Code via Semantic and Collaborative Integration](https://arxiv.org/abs/2502.06269)      | TOIS'25 |                /
| *RecGPT* |      [RecGPT: A Foundation Model for Sequential Recommendation](https://arxiv.org/abs/2506.06270)      | EMNLP'25 |                 [![Github](https://img.shields.io/github/stars/hkuds/recgpt.svg?style=social&label=Github)](https://github.com/hkuds/recgpt)   
| *MMQ* |      [MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation](https://arxiv.org/abs/2508.15281)      | arXiv'25 |                /
| *MME-SID* |      [Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs](https://arxiv.org/abs/2509.02017)      | CIKM'25 |                 [![Github](https://img.shields.io/github/stars/Applied-Machine-Learning-Lab/MME-SID.svg?style=social&label=Github)](https://github.com/Applied-Machine-Learning-Lab/MME-SID)   
| *GNPR-SID* |      [Generative Next POI Recommendation with Semantic ID](https://arxiv.org/abs/2506.01375)      | KDD'25 |                 [![Github](https://img.shields.io/github/stars/wds1996/GNPR-SID.svg?style=social&label=Github)](https://github.com/wds1996/GNPR-SID)     
| *COBRA* |      [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations](https://arxiv.org/abs/2503.02453)      | arXiv'25 |                /  
| *GFlowGR* |      [GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks](https://arxiv.org/abs/2506.16114)      | arXiv'25 |                /  
| *STREAM-Rec* |      [Slow Thinking for Sequential Recommendation](https://arxiv.org/abs/2504.09627)      | arXiv'25 |                /  
| *AtSpeed* |      [Efficient Inference for Large Language Model-based Generative Recommendation](https://arxiv.org/abs/2410.05165)      | ICLR'25 |                 [![Github](https://img.shields.io/github/stars/Linxyhaha/AtSpeed.svg?style=social&label=Github)](https://github.com/Linxyhaha/AtSpeed) 
| *RecBase* |      [RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation](https://arxiv.org/abs/2509.03131)      | EMNLP'25 |                /  
| *GMC* |      [Generative Multi-Target Cross-Domain Recommendation](https://arxiv.org/abs/2507.12871)      | arXiv'25 |                /  
| *SaviorRec* |      [SaviorRec: Semantic-Behavior Alignment for Cold-Start Recommendation](https://arxiv.org/abs/2508.01375)      | arXiv'25 |                /  
| *TALKPLAY* |      [TALKPLAY: Multimodal Music Recommendation with Large Language Models](https://arxiv.org/abs/2502.13713)      | arXiv'25 |                /  
| *CAR* |      [Act-With-Think: Chunk Auto-Regressive Modeling for Generative Recommendation](https://arxiv.org/abs/2506.23643)      | arXiv'25 |                /  
| *QARM* |      [QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739)      | CIKM'25 |                /  
| *EGA-V2* |      [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549)      | arXiv'25 |                /  
| *PLUM* |      [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784)      | arXiv'25 |                /  
| *OneRec-Think* |      [OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639)      | arXiv'25 |                /  
| *SpecGR* |      [Inductive Generative Recommendation via Retrieval-based Speculation](https://arxiv.org/abs/2410.02939)      | AAAI'26 |                 [![Github](https://img.shields.io/github/stars/Jamesding000/SpecGR.svg?style=social&label=Github)](https://github.com/Jamesding000/SpecGR) 



### Architecture Design
Generative recommendation systems often utilize encoder-decoder and decoder-only architectures. These structures enable scalability and better computational efficiency compared to traditional methods. The evolution of architectures from simple MLP models to large transformer-based models is covered.

### Optimization Strategies
Optimization plays a crucial role in enhancing the effectiveness of generative recommendation systems. We discuss multi-objective optimization strategies that balance user satisfaction, computational efficiency, and business objectives.

## Applications

### Industrial Use Cases
Generative recommendation systems are rapidly being deployed in various industries:
- **E-commerce**: Personalized product recommendations.
- **Streaming Services**: Suggesting movies, music, and content.
- **Healthcare**: Personalized treatment plans based on historical data.
- **Education**: Adaptive learning paths for students.

### Challenges and Future Directions
Despite the promising capabilities of generative models, there remain several challenges:
- **Cold-start Problems**: How to efficiently handle new users or items with limited data.
- **Scalability**: Managing large datasets with computational efficiency.
- **Ethical Considerations**: Balancing personalization with privacy concerns.



## Acknowledgement
This research was conducted as a collaboration between City University of Hong Kong and Kuaishou Technology. The authors thank Kuaishou Technology for providing data support and technical resources, and City University of Hong Kong for theoretical guidance and academic supervision. We also acknowledge the computational resources and experimental environments provided by both institutions.

[![City University of Hong Kong](figures/City_University_of_Hong_Kong_(2024).svg.png)](https://www.cityu.edu.hk/)
[![Kuaishou Technology](figures/Kuaishou_logo_(2020).png)](https://www.kuaishou.com/)

## How to Contribute
We welcome contributions from the community! If you have suggestions, improvements, or want to add papers to the reading list, feel free to submit an issue on [GitHub Issues](https://github.com/Kuaishou-RecModel/Tri-Decoupled-GenRec/issues).

<!-- ## Contribute
If you find this survey useful, please cite the following paper:

```bibtex
@misc{your2025generative,
  title={Generative Recommendation Systems: A Comprehensive Survey},
  author={Your Name and Collaborators},
  year={2025},
  eprint={your-arxiv-id},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
}
``` -->

<!-- ## Authors
- **Your Name** (Your Affiliation)
- **Collaborator 1** (Affiliation)
- **Collaborator 2** (Affiliation) -->
