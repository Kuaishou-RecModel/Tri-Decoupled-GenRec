
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
| *HSTU* |      [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)      | ICML'24 |                        [![Github](https://img.shields.io/github/stars/meta-recsys/generative-recommenders.svg?style=social&label=Github)](https://github.com/meta-recsys/generative-recommenders)                        |
| *MTGR* |      [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654)      | CIKM '25 |                        /                        |
| *PinRec* |      [PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems](https://arxiv.org/abs/2504.10507)      | arXiv '25 |                        /                        |

- **Text-based Tokenization:** Leverages the power of natural language models, improving semantic understanding.
- **Semantic ID-based Tokenization:** Combines the best of both worlds with efficient, semantic-rich representations.






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
