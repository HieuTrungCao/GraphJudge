import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from datasets import load_dataset
import transformers
# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
import argparse

import wandb

# Login to W&B. You can use Kaggle secrets for security.
# The API key can be found in your W&B profile settings.
from kaggle_secrets import UserSecretsClient
try:
    wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
except:
    wandb.login() # If running locally or without Kaggle secrets, this will prompt for the key

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 8                                            # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128                                               # 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2                                                      # we don't always need 3 tbh
STEPS = 500
LEARNING_RATE = 3e-4                                            # the Karpathy constant
CUTOFF_LEN = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# DATA_PATH = "data/genwiki_4omini_context/train_instructions_context_llama2_7b.json"
# OUTPUT_DIR = "models/llama2-7b-lora-genwiki-context"
base_model_path = "NousResearch/Llama-2-7b-hf"
# base_model_path = "meta-llama/Llama-2-7b"


# ddp
device_map = "cuda"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# model
model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map=device_map,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False
model.config.pretraining_tp = 1

def generate_and_tokenize_prompt(data_point):
    # 确保输入数据是字符串类型
    instruction = str(data_point["instruction"])
    input_text = str(data_point["input"]) if data_point["input"] else ""
    output = str(data_point["output"])

    # 构建提示文本
    if input_text:
        user_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input_text}
### Response:
"""
    else:
        user_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
"""

    encoded = tokenizer(
        user_prompt + output,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors=None,
    )

    # 编码提示部分以获取其长度
    prompt_encoded = tokenizer(
        user_prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )

    labels = [-100] * len(prompt_encoded["input_ids"])
    labels.extend(encoded["input_ids"][len(prompt_encoded["input_ids"]):])
    
    if len(labels) < CUTOFF_LEN:
        labels.extend([-100] * (CUTOFF_LEN - len(labels)))
    labels = labels[:CUTOFF_LEN]

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--finput", type=str, default="data/WN11/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN11/pred_instructions_llama2_7b.csv")
    # parser.add_argument("--finput", type=str, default="data/FB13/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/FB13/pred_instructions_llama2_7b.csv")
    parser.add_argument("--data_path", type=str, default="data/genwiki_4omini_context/train_instructions_context_llama2_7b.json")
    parser.add_argument("--output_dir", type=str, default="ueihieu/llama2-7b-lora-genwiki")  # pred_instructions_context_genwiki_llama2_7b_itr1.csv
    parser.add_argument("--model_hub", type=str, default="")  # pred_instructions_context_genwiki_llama2_7b_itr1.csv
    # parser.add_argument("--finput", type=str, default="data/WN18RR/test_instructions_llama_merge.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN18RR/pred_instructions_llama2_7b_merge.csv")
    _args = parser.parse_args()

    # 加载数据集
    data = load_dataset("json", data_files=_args.data_path)
    
    # 数据预处理
    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
    else:
        train_data = (
            data["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        val_data = None
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            # per_device_train_batch_size=MICRO_BATCH_SIZE,
            # gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            max_steps=STEPS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=5,
            eval_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            eval_steps=20 if VAL_SET_SIZE > 0 else None,
            save_steps=20,
            output_dir=_args.output_dir,
            save_total_limit=3,
            hub_model_id=_args.model_hub,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            optim="adamw_torch",
            report_to="wandb",
            logging_strategy="steps",
            logging_first_step=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )

    print
    # 训练模型
    trainer.train()

    trainer.push_to_hub()
    # 保存模型和分词器
    model.save_pretrained(_args.output_dir)
    tokenizer.save_pretrained(_args.output_dir)
    print("\n Training completed! Model saved to:", _args.output_dir)