# Evaluating LLMs on the WiC Task
The code and data relevant to the COLING 2025 paper are maintained in this repository.
- Yoshihiko Hayashi, "Evaluating LLMs' Capability to Identify Lexical Semantic Equivalence: Probing with the Word-in-Context Task." [COLING2025](https://coling2025.org/)

## Quick overview

* **Tested environment**
  * Python 3.8.10
  * openai 0.27.7
  * ray 2.6.2
  * Other standard libraries such as numpy, pandas, sklearn, etc.

* Contact mailto:yoshihiko.hayashi@gmail.com

***

## To reproduce the results ##
You need to collect zero-shot results.

## Directories
* **WiC_dataset**: This directory maintains the original Word-in-Context dataset. Please refer to the original WiC paper for details.
  * M.T. Pilehvar and J. Camacho-Collados, [WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations.](https://aclanthology.org/N19-1128/), NAACL 2019.
  * https://pilehvar.github.io/wic/
  
## Python scripts
* **repair_tword_index.py**: This script repairs the tokenization of contextual sentences in the original WiC files. A usage example is "$ python repair_tword_index.py train", which will produce a train_new.data.txt file in the WiC_dataset_repaired directory from the original train.data.txt file.

* **zero_shot.py**: This script is prepared for obtaining zero-shot baseline results. A usage example is "$ python zero_shot.py --llm gpt-4-0613", which executes a zero-shot run on the test dataset using GPT-4. The prompt template is coded in this file, and the adjective to dictate the degree of semantic sameness (default is "identical") can be altered using the "adj" argument.

* **Other files**: anal_desc.py and anaysis_tool.py can be used for replicating the additional results reported in the paper. 

***
## Disclaimer
The code and data in this repository are provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
