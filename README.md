# Aspect-Based Opinion Extraction from Restaurant Reviews
## NLP Course @CentraleSupélec — 2025-2026
 
---
 
## Authors
 
- Hala Chafik
- Farouk Yartaoui
- Hamza Tbatou
 
---
 
## Project Description
 
### Task
 
The goal of this project is to extract aspect-based opinions from French customer reviews about restaurants. For each review, the system predicts the overall sentiment expressed toward each of 3 predefined aspects: **Price**, **Food**, and **Service**. Each aspect is assigned one of 4 possible labels: Positive, Negative, Mixed, or No Opinion.
 
### Approach: LoRA Fine-Tuning of a Causal Language Model (Approach 2)
 
We chose to reframe the classification task as a **text generation task** using a causal (decoder-only) large language model fine-tuned with LoRA (Low-Rank Adaptation). Rather than training a dedicated classification head, the model is prompted with a structured instruction describing the task and must generate the 3 labels in a fixed format. This design leverages the strong multilingual pretraining of modern LLMs, including their comprehension of French text, while only requiring a small number of additional trainable parameters to adapt to the task.
 
### Base Model
 
We used **Qwen/Qwen3-1.7B**, a 1.7 billion parameter multilingual causal language model from Alibaba's Qwen3 family. It was selected because it fits within the 32GB VRAM constraint when loaded in fp16 precision, offers strong French language understanding. The model is loaded with `dtype=torch.float16` and `device_map=None` to allow Distributed Data Parallel (DDP) training via `accelerate`.
 
### Prompt Design
 
Each training sample is formatted as a prompt-completion pair. The prompt provides the task description, aspect definitions, and the review text. The completion contains the ground-truth labels in a structured format:
 
```
Price: <label>
Food: <label>
Service: <label>
```
 
The prompt and completion are concatenated and fed to the SFT trainer. The loss is computed only on the completion tokens.
 
### LoRA Configuration
 
Full parameter fine-tuning of 1.7B parameters would be expensive and prone to overfitting on a dataset of this size. We therefore applied LoRA, which freezes the base model weights and introduces low-rank trainable matrices into the attention layers. Our configuration uses rank `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, and targets the `q_proj` and `v_proj` attention projection matrices. This results in only **3,211,264 trainable parameters out of 1,723,786,240 total (0.19%)**.
 
### Training Setup
 
Training was performed using HuggingFace TRL's `SFTTrainer` for 3 epochs with fp16 mixed precision. The effective batch size is set to 16 and scales automatically with the number of available GPUs via gradient accumulation. The learning rate is set to `2e-4` and scales linearly with the effective batch size.
 
### Inference
 
At prediction time, the base model is reloaded and the LoRA adapter is applied via `PeftModel.from_pretrained()`. For each review, the prompt is constructed and the model generates up to 30 new tokens using greedy decoding (`do_sample=False`). The generated output is parsed with a regular expression to extract the predicted label for each aspect. If a label is missing or unrecognized, it defaults to No Opinion.
 
---
 
## Results on the Development Set
 
The program was evaluated over 3 runs on the full development set. Results are as follows:
 
| Run | Price | Food | Service | Macro Avg Accuracy |
|-----|-------|------|---------|-------------------|
| 1   | 87.67% | 85.17% | 87.00% | **86.61%** |
| 2   | 87.67% | 85.00% | 87.50% | **86.72%** |
| 3   | 87.5% | 85.15% | 87.50% | **86.72%** |
 
**Average Macro-Average Accuracy across runs: 86.68%**
 
---
 
## How to Run
 
From the `src` directory:
 
```bash
accelerate launch runproject.py
```
