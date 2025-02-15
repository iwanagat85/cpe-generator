import datetime
import os
from os import path

import pandas
import torch
from datasets import Dataset
from peft import LoraConfig, PeftType, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig, \
    SchedulerType, IntervalStrategy
from transformers.trainer_utils import SaveStrategy
from transformers.training_args import OptimizerNames
from trl import SFTTrainer, SFTConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_path = "datas/categorized_cpes_150000.csv"
output_dir = "outputs"
logging_dir = "logs"

now = datetime.datetime.now()

DEFAULT_SYSTEM_PROMPT = """
Generate a JSON from the given text.
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
```json
{
	"part": string  // May have 1 of 3 values. a for Applications. h for Hardware. o for Operating Systems.
	"vendor": string  // Values for this attribute SHOULD describe or identify the person or organization that manufactured or created the product.
	"product": string  // The name of the system/package/component. product and vendor are sometimes identical.
	"version": string  // The version of the system/package/component.
}
```"""


def get_logging_dir():
    now_str = now.strftime("%Y%m%d%H%M%S")
    return path.join(logging_dir, now_str, "network_train")


def get_output_dir():
    now_str = now.strftime("%Y%m%d%H%M%S")
    return path.join(output_dir, now_str)


def load_train_dataset() -> Dataset:
    names = ['title', 'part', 'vendor', 'product', 'version']
    df = pandas.read_csv(dataset_path, header=0, names=names)
    return Dataset.from_pandas(df)


def formatting_prompts_func(examples, tokenizer):
    texts = []
    for title, part, vendor, product, version in zip(
            examples["title"],
            examples["part"],
            examples["vendor"],
            examples["product"],
            examples["version"],
    ):
        content = f"""```json
        {{
            "part": "{part}",
            "vendor": "{vendor}",
            "product": "{product}",
            "version": "{version}"
        }}
        ```"""
        chat_template = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": title},
            {"role": "assistant", "content": content},
        ]
        text = tokenizer.apply_chat_template(
            chat_template,
            tokenize=False,
            add_generation_prompt=False,
            max_length=512,
        )
        texts.append(text)
    return {"text": texts}


def load_model_and_tokenizer():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        use_fast=True,
    )
    return model, tokenizer


def get_peft_config():
    return LoraConfig(
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        peft_type=PeftType.LORA,
        task_type=TaskType.CAUSAL_LM,
    )


def get_training_args():
    return SFTConfig(
        output_dir=get_output_dir(),
        per_device_train_batch_size=6,
        gradient_accumulation_steps=18,
        learning_rate=0.00001,
        max_seq_length=512,
        weight_decay=0.1,
        max_grad_norm=0.01,
        num_train_epochs=1,
        lr_scheduler_type=SchedulerType.COSINE,
        warmup_ratio=0.05,
        logging_dir=get_logging_dir(),
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=10,
        save_strategy=SaveStrategy.STEPS,
        save_steps=100,
        save_total_limit=10,
        save_safetensors=True,
        seed=1234,
        bf16=True,
        dataloader_num_workers=4,
        optim=OptimizerNames.PAGED_ADAMW_8BIT,
        # optim_args=,
    )


def main():
    print(os.getcwd())
    print(f"device: {device}")

    model, tokenizer = load_model_and_tokenizer()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(model)

    train_dataset = load_train_dataset()
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )

    peft_config = get_peft_config()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_arguments = get_training_args()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
        args=training_arguments,
    )
    trainer.train()
    trainer.save_model(get_output_dir())
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
