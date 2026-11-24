import math
import time
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

MODEL_NAME = "distilgpt2"
CONTEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
print(f"parameters {MODEL_NAME}: {model.num_parameters():,}")

raw_ds = load_dataset(
    "text",
    data_files={"train": "data/train_raw.txt", "validation": "data/val_raw.txt"},
    sample_by="document",
)


def tokenize_fn(examples):
    return tokenizer(examples["text"])


tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // CONTEXT_LEN) * CONTEXT_LEN
    result = {
        k: [v[i:i + CONTEXT_LEN] for i in range(0, total_len, CONTEXT_LEN)]
        for k, v in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized.map(group_texts, batched=True, remove_columns=tokenized["train"].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="out/distilgpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
)

t0 = time.time()
trainer.train()
train_time = time.time() - t0

eval_metrics = trainer.evaluate()
val_loss = eval_metrics["eval_loss"]
val_ppl = math.exp(val_loss)

print(f"\nfine-tuned {MODEL_NAME}:")
print(f"  val_loss: {val_loss:.4f}")
print(f"  val_ppl:  {val_ppl:.2f}")
print(f"  train time: {train_time / 60:.1f} min")

trainer.save_model("out/distilgpt2-finetuned/final")

model.eval()
prompts = ["ROMEO:", "To be, or not to be", "KING RICHARD:"]
with open("out/distilgpt2_generations.txt", "w", encoding="utf-8") as f:
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out = model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        f.write(f"=== prompt: {prompt!r} ===\n{text}\n\n")
        print(f"\n=== {prompt!r} ===\n{text[:300]}")
