
# SafeLawBench: Towards Safe Alignment of Large Language Models

| [**📖 Paper**](https://arxiv.org/abs/2506.06636) |



## Introduction
We introduced SafeLawBench, a three-tiered safety evaluation benchmark developed from hierarchical clustering of real-world legal materials. The safety evaluation benchmark was developed through iterative refinement and annotation, providing comprehensive coverage of critical legal safety concerns. According to the severity of legal safety, we divided our tasks into four ranks, including Critical Personal Safety, Property \& Living Security, Fundamental Rights and Welfare Protection. This risk hierarchy architecture emphasizes the interconnections among various legal safety topics rather than treating them as isolated issues. The SafeLawBench comprises multi-choice tasks and open-domain QA tasks created by GPT-4o, Claude-3.5-Sonnet and Gemini based on public legal materials. Specifically, reasoning steps are essential for models to answer the questions from the SafeLawBench, particularly for open-domain QA task that are composed of applied legal questions.

![logo](./imgs/architecture.png)


## Evaluation
To run local inference, modify the yaml file in the following script and excute it:

```bash
cd scripts
sh run_inference.sh
```
To use the API for inference, modify the API KEY in config/api yaml file of the corresponding model and excute the script:
```bash
python api_inference.py --config_file /hklexsafe/configs/inference/api/deepseek-r1.yaml
```

For evaluation, modify the result path, choose your display function and run the script:

```bash
cd safelawbench/evaluation
python LexSafeEvaluator.py
```

## Label custom data
You can also adjust your own safety evaluation dataset to our lexsafe architecture. Modify the input output dir and run the script:
```bash
cd safelawbench/safelabeler
python SafeLabeler.py
```
Before evaluation, add the `classification_keys.json`.

## 🏆 Mini-Leaderboard

### Multi-choice questions


|Model|Open?|Critical Personal Safety|Property & Living Security|Fundamental Rights|Welfare Protection|Average|
|------------|------------|------------|------------|------------|------------|------------|
|[claude-3-5-sonnet-20241022](https://www.anthropic.com/news/claude-3-5-sonnet)|No|82.4|79.6|**80.0**|**79.8**|**80.5**|
|[gpt-4o](https://openai.com/index/hello-gpt-4o/)|No|**83.2**|**79.9**|79.3|78.8|80.3|
|[deepseek-v3-0324]()|Yes|82.9|79.2|78.3|79.1|79.7|
|[deepseek-r1]()|Yes|81.4|77.9|77.1|77.8|78.5|
|[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)|Yes|81.4|76.5|76.3|76.5|77.6|
|[Mistral-Large-Instruct-2411](https://huggingface.co/models/Mistral-Large-Instruct-2411)|Yes|81.2|75.3|76.5|76.2|77.2|
|[Meta-Llama-3-70B-Instruct](https://huggingface.co/models/Llama-3-70B-Instruct)|Yes|79.9|74.6|75.1|74.8|76.1|
|[QwQ-32B]()|Yes|79.3|74.3|74.5|74.6|75.6|
|[Llama-3.1-70B-Instruct](https://huggingface.co/models/Llama-3.1-70B-Instruct)|Yes|78.5|74.4|74.0|74.5|75.2|
|[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)|Yes|78.8|73.2|73.4|75.0|74.9|
|[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)|Yes|74.9|69.4|69.5|70.7|70.9|
|[gemma-2-27b-it](https://huggingface.co/models/Gemma-2-27B-IT)|Yes|76.0|68.6|68.7|69.0|70.5|
|[Mistral-Small-Instruct-2409](https://huggingface.co/models/Mistral-Small-Instruct-2409)|Yes|72.9|67.9|67.0|68.3|68.8|
|[Meta-Llama-3-8B-Instruct](https://huggingface.co/models/Llama-3-8B-Instruct)|Yes|71.1|68.3|66.7|68.5|68.4|
|[Llama-3.1-8B-Instruct](https://huggingface.co/models/Llama-3.1-8B-Instruct)|Yes|68.8|64.5|63.8|64.3|65.3|
|[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)|Yes|66.3|60.7|61.3|61.9|62.4|
|[glm-4-9b-chat](https://huggingface.co/models/GLM-4-9B-Chat)|Yes|64.7|60.0|59.8|60.9|61.2|
|[gemma-2-2b-it](https://huggingface.co/models/Gemma-2-2B-IT)|Yes|63.2|57.1|57.2|57.6|58.7|
|[vicuna-7b-v1.5](https://huggingface.co/models/Vicuna-7B-V1.5)|Yes|48.7|43.8|44.2|43.0|45.1|
|[vicuna-13b-v1.5](https://huggingface.co/models/Vicuna-13B-V1.5)|Yes|33.4|29.0|29.2|28.0|30.0|


### Open-domain QAs

| Model                             | Elo Score |
|-----------------------------------|-------|
| DeepSeek-R1                       | 5651  |
| Qwen2.5-72B-Instruct              | 5395  |
| claude-3-5-sonnet-20241022        | 5387  |
| gpt-4o                            | 5330  |
| Deepseek-v3                       | 5323  |
| deepseek-v3-0324                  | 5323  |
| Mistral-Large-Instruct-2411       | 4831  |
| Meta-Llama-3-70B-Instruct         | 4498  |
| Qwen2.5-14B-Instruct              | 4441  |
| Llama-3.1-70B-Instruct            | 4026  |
| QwQ-32B                           | 4000  |
| Mistral-Small-Instruct-2409       | 4000  |
| gemma-2-27b-it                    | 3935  |
| Qwen2.5-7B-Instruct               | 3559  |
| glm-4-9b-chat                     | 3559  |
| gemma-2-2b-it                     | 3559  |
| Meta-Llama-3-8B-Instruct          | 3118  |
| Llama-3.1-8B-Instruct             | 2677  |
| Qwen2.5-3B-Instruct               | 2236  |
| vicuna-13b-v1.5                   | 1795  |
| vicuna-7b-v1.5                    | 1354  |


## Contact
- Chuxue Cao: ccaoai@connect.ust.hk


## Citation

**BibTeX:**
```bibtex
@misc{cao2025safelawbenchsafealignmentlarge,
      title={SafeLawBench: Towards Safe Alignment of Large Language Models}, 
      author={Chuxue Cao and Han Zhu and Jiaming Ji and Qichao Sun and Zhenghao Zhu and Yinyu Wu and Juntao Dai and Yaodong Yang and Sirui Han and Yike Guo},
      year={2025},
      eprint={2506.06636},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.06636}, 
}
```
