<div align="center">

<img src="assets/sii.jpg" alt="SII" width="96" height="96">
<img src="assets/asi.png" alt="ASI" width="96" height="96">

# LIMI: Less is More for Agency

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2509.17567" target="_blank">Paper</a> &nbsp; | &nbsp;
  🌐 <a href="https://huggingface.co/datasets/GAIR/LIMI" target="_blank">Dataset </a> &nbsp; | &nbsp;
  📘 <a href="https://huggingface.co/GAIR/LIMI" target="_blank">Model </a>
</p>


<p align="center"> <img src="./assets/teaser.jpg" style="width: 85%;" id="title-icon">       </p>
</div>


## 📌 Table of Contents
- [LIMI: Less is More for Agentic Intelligence 🚀](#limi-less-is-more-for-agentic-intelligence)
  - [Updates](#updates)
  - [📌 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Model Zoo](#model-zoo)
  - [Datasets](#datasets)
  - [Quick Start](#quick-start)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [License](#license)
  - [Citation](#citation)


## Overview

LIMI is a framework that explores this possibility by demonstrating that strategic data curation and focused development of essential agentic capabilities can yield superior agentic systems with significantly reduced computational requirements. 

## Model Zoo

Our LIMI models are available on Hugging Face 🤗:

| Model | Backbone | Size | Link |
|-------|----------|------|------|
| **LIMI** | [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) | 355B | [🤗](https://huggingface.co/GAIR/LIMI) |
| **LIMI-Air** | [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) | 106B | [🤗](https://huggingface.co/GAIR/LIMI-Air) |


## Datasets

We release our datasets through Hugging Face 🤗:

| Dataset | Description | Link |
|---------|-------------|------|
| **LIMI** | Updated training set for the paper (78 samples) | [🤗](https://huggingface.co/datasets/GAIR/limi) |

## Quick Start

Our models are fine-tuned on [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) and are compatible with most mainstream frameworks like [HF Transformers](https://github.com/huggingface/transformers), [SGLang](https://github.com/sgl-project/sglang), [Megatron](https://github.com/NVIDIA/Megatron-LM), [slime](https://github.com/THUDM/slime) and etc. 

### Using the Latest Model (LIMI)

<details>
<summary>Start with HF Transformers</summary>

```bash
# Install required packages
pip install transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "GAIR/LIMI",
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("GAIR/LIMI", trust_remote_code=True)

# Prepare input messages (We use the following template and system prompt during training and inference)
messages = [
    {"role": "system", "content": "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems."},
    {"role": "user", "content": "Modify the \texttt{equation.py} function, considering the physical meaning and relationships of the inputs."}
]

# Format input using chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize input
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=128000,
    temperature=0.6,
    top_p=0.95,
    do_sample=True
)

# Decode and print response
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

</details>

<details>
<summary>Start with VLLM</summary>

```bash
# Install required packages
pip install vllm
```


```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Initialize the model
llm = LLM(
    model="GAIR/LIMI",
    tensor_parallel_size=4,  # adjust based on available GPUs
    trust_remote_code=True,
    swap_space=60,
    gpu_memory_utilization=0.96,
)

# Prepare input messages (We use the following template and system prompt during training and inference)
messages = [
    {"role": "system", "content": "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems."},
    {"role": "user", "content": "Modify the \texttt{equation.py} function, considering the physical meaning and relationships of the inputs."}
]

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("GAIR/LIMI", trust_remote_code=True)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Configure generation parameters
sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=128000,
    top_p=0.95,
)

# Generate response
output = llm.generate(text, sampling_params)
print(output[0].outputs[0].text)
```

</details>


## Training

We utilize [slime](https://github.com/THUDM/slime) framework for training, which provides a convenient and efficient training pipeline.

1. **Environment Setup**
   - Set up slime following their official [documentation](https://github.com/THUDM/slime).
   - Ensure all dependencies are properly installed and configured.

2. **Data Preparation**
   - Obtain the LIMI dataset from [🤗 Hugging Face](https://huggingface.co/datasets/GAIR/LIMI).

3. **Configuration**
   - Use our provided [training script](https://github.com/GAIR-NLP/LIMI/scripts/train/train_glm4.5.sh).
   - The script file contains all necessary hyperparameters and training settings.


## Evaluation

To support the rigorous assessment of agentic capabilities outlined in this work, we release a comprehensive evaluation suite. This framework is designed to benchmark Large Language Models (LLMs) on the held-out evaluation subset $D_{\text{eval}}$ and is structured around two core components: an inference module (utilizing the VLLM framework for efficient generation) and an evaluation module for scoring model performance against our defined metrics.

The evaluation module implements the four key metrics: First-Turn Functional Completeness (FTFC), Success Rate (SR@R), Remaining Chances (RC@R), with a computational budget of R = 3 rounds. For FTFC and other rule-based assessments, we employ precise matching of requirements and outcomes.

For detailed instructions on usage, implementation details, and replication of our experimental results, please refer to the complete documentation available at \texttt{eval/README.md}.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{xiao2025limiagency,
      title={LIMI: Less is More for Agency}, 
      author={Yang Xiao and Mohan Jiang and Jie Sun and Keyu Li and Jifan Lin and Yumin Zhuang and Ji Zeng and Shijie Xia and Qishuo Hua and Xuefeng Li and Xiaojie Cai and Tongyu Wang and Yue Zhang and Liming Liu and Xia Wu and Jinlong Hou and Yuan Cheng and Wenjie Li and Xiang Wang and Dequan Wang and Pengfei Liu},
      year={2025},
      eprint={2509.17567},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.17567}, 
}
```
