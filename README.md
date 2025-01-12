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

## To reproduce the reported results ##
* First, collect zero-shot results. Use zero_shot_gpt.py to invoke GPT models and zero_shot_{llama, mistral}.py for Llama and Mistral models. The `zs_tsv` directory stores the result tab-separated files. The naming convention for these files is `{LLM_name}_{Adjective}_{Data_split}_{Run_ID}.tsv`. Note that the `Run_IDs` are currently fixed to `323`.
    * Remember to set your OpenAI API key in the environment variable to use the GPT models.
* Table 1: You can use the `for_paper` function in `read_summ.py`.
* Table 2 and 3: The `for_paper` function in `fleiss2.py` can be used. Specify `fleiss2.predictor_comb_set` as the argument to replicate Table 2 and `fleiss2.predictor_adj_set` to reproduce Table 3.
* Table 4: Use the `comp_for_adjs` function in `read_summ.py`, specifying an LLM name, such as `Llama3-8B`, as the argument.
* Table 5: Use the `main` function in `rank_corr.py` with the LLM name as the argument to generate the results shown in Table 5. Ensure the corresponding LLM-specific tab-separated file is located in the `llms_pr` directory.
* Table 6: This table extracts relevant information obtained by the `main__` function in `compare_res.py`. You need to specify the dataset (`train`, `dev`, or `test`) as the argument.
* Instances manually inspected: These WiC instances are extracted into tab-separated files in the inspected_instances directory. They are derived from the zero-shot results obtained by the GPT-4o/the same predictor.
* Experiments on ensembling predictors: The `main` function in the `greedy_ensemble.py` file handles ensembling experiments. You can specify a combination of predictors (with some standard combinations defined in the file) as the argument. The `greedy_ensemble_results` directory stores the `.pkl` files summarizing the major results. You can use the `show_` function to read them.

## WiC_dataset 
* The `WiC_dataset` directory contains the original Word-in-Context dataset. Please refer to the original WiC paper and the corresponding website for further details.
  * M.T. Pilehvar and J. Camacho-Collados, [WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations.](https://aclanthology.org/N19-1128/), NAACL 2019.
  * https://pilehvar.github.io/wic/
  
***
## Disclaimer
The code and data in this repository are provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
