# Spanish-MaliciousHateSpeech
## Detection of Maliciously Disseminated Hate Speech in Spanish Using Fine-Tuning and In-Context Learning Techniques with Large Language Models  
*(Accepted in CMC — Computers, Materials & Continua, 2025)*

### TL-DR: Highlights
- This repository provides the **implementation and experimental framework** from the paper.
- The study is conducted **on top of the [Spanish-MTLHateCorpus 2023](https://github.com/NLP-UMUTeam/Spanish-MTLHateCorpus)** dataset:
- We evaluate the detection of **maliciously disseminated hate speech**, a challenging subtype requiring more complex semantic and pragmatic reasoning.
- Comprehensive comparison between:
  - **Fine-tuning** approaches  
  - **In-Context Learning (ICL)** with instruction-based prompting  
  - **Zero-shot and few-shot** LLM settings  
- The paper demonstrates that ICL can approach fine-tuning performance in certain subtasks, while offering practical advantages for rapid prototyping and low-resource scenarios.

### Authors

- **Tomás Bernal-Beltrán** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=0bTUxQEAAAAJ&hl=en) · [ORCID](https://orcid.org/0009-0006-6971-1435)

- **Ronghao Pan** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=80lntLMAAAAJ) · [ORCID](https://orcid.org/0009-0008-7317-7145)

- **José Antonio García-Díaz** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=ek7NIYUAAAAJ) · [ORCID](https://orcid.org/0000-0002-3651-2660)

- **María del Pilar Salas-Zárate** — Tecnológico Nacional de México / ITS Teziutlán  
  [Google Scholar](https://scholar.google.com/citations?user=2ssaDdsAAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0003-1818-3434)

- **Mario Andrés Paredes-Valverde** — Tecnológico Nacional de México / ITS Teziutlán  
  [Google Scholar](https://scholar.google.com/citations?user=AYJZ7cEAAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0001-9508-9818)

- **Rafael Valencia-García** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=GLpBPNMAAAAJ) · [ORCID](https://orcid.org/0000-0003-2457-1791)  
  **Corresponding author:** valencia@um.es

> **Affiliations:**  
> \* *Departamento de Informática y Sistemas, Universidad de Murcia, Campus de Espinardo, Murcia, Spain*  
> \* *Tecnológico Nacional de México / ITS Teziutlán, Puebla, Mexico*


### Publication
This work has been accepted for publication in **CMC — Computers, Materials & Continua (2025)**.  
More information (DOI, volume, issue) will be added upon release. 


### Abstract
The malicious dissemination of hate speech via compromised accounts, automated bot networks, and malware-driven social media campaigns has become a growing cybersecurity concern. Automatically detecting such content in Spanish is challenging due to linguistic complexity and the scarcity of annotated resources. 

In this paper, we compare two predominant AI-based approaches for the forensic detection of malicious hate speech:  
(1) **fine-tuning encoder-only Spanish LLMs**, and  
(2) **In-Context Learning (ICL)** techniques (Zero- and Few-Shot Learning) with large-scale language models.

Our approach goes beyond binary classification, proposing a comprehensive multidimensional evaluation that labels each text according to:  
1. **Type of speech**,  
2. **Recipient**,  
3. **Level of intensity** (ordinal), and  
4. **Targeted group** (multi-label).

Performance is evaluated using an annotated Spanish corpus and standard metrics (precision, recall, F1-score) as well as stability-oriented metrics to quantify the transition from zero-shot to few-shot prompting, namely **Zero-to-Few Shot Retention** and **Zero-to-Few Shot Gain**.

Results show that fine-tuned encoder-only models—particularly **MarIA** and **BETO variants**—consistently achieve the strongest and most reliable performance, with macro F1-scores roughly in the **46–66%** range depending on the task. Zero-shot approaches are considerably less stable and typically obtain substantially lower performance (**0–39% F1**), often producing invalid outputs. Few-shot prompting (e.g., Qwen 3 8B, Mistral 7B) improves stability and recall relative to zero-shot, reaching **20–51% F1**, but still falls short of fully fine-tuned models.

These findings highlight the importance of supervised adaptation and discuss the potential of both paradigms as components in AI-powered cybersecurity and malware forensic systems aimed at identifying and mitigating coordinated online hate campaigns.


## Acknowledgments
This work is part of the research project **LaTe4PoliticES (PID2022-138099OB-I00)** funded by *MCIN/AEI/10.13039/501100011033* and the *European Fund for Regional Development (ERDF)-a way to make Europe*. Mr. Tomás Bernal-Beltrán is supported by **University of Murcia** through the *predoctoral programme*.


#### Citation
A complete BibTeX citation will be added when the DOI is assigned.

```
@article{bernal2025malicioushatespeech,
  title={Detection of Maliciously Disseminated Hate Speech in Spanish Using Fine-Tuning and In-Context Learning Techniques with Large Language Models},
  author={Bernal-Beltr{\'a}n, Tom{\'a}s and Pan, Ronghao and Garc{\'\i}a-D{\'\i}az, Jos{\'e} Antonio and Salas-Z{\'a}rate, Mar{\'\i}a del Pilar and Paredes-Valverde, Mario Andr{\'e}s and Valencia-Garc{\'\i}a, Rafael},
  journal={CMC -- Computers, Materials \& Continua},
  year={2025},
  note={Accepted}
}

```



