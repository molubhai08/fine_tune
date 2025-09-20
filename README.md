# fine_tune

# Fine-tuning a Hugging Face Model

This project demonstrates how to fine-tune a Hugging Face Transformer model on a custom dataset using the `Trainer` API. It covers preprocessing, training, evaluation, and logging.

---

## 🚀 Features

* Uses Hugging Face `Trainer` for training and evaluation
* Handles dataset tokenization & dynamic padding with `data_collator`
* Computes evaluation metrics during training
* Compatible with Hugging Face Datasets (`train` and `validation` splits)
* **No external logging required** (W\&B is disabled by default)

---

## 📦 Requirements

Install dependencies:

```bash
pip install transformers datasets evaluate
```

(Optional, for TensorBoard visualization):

```bash
pip install tensorboard
```

---

## ⚙️ Training Script

The key part of the notebook is creating and running the `Trainer`:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,   # replaces deprecated `tokenizer`
    data_collator=data_collator,  # dynamically pads batches
    compute_metrics=compute_metrics,
    report_to=[]                  # disables Weights & Biases
)

trainer.train()
```

---

## 📊 Logging

* By default, W\&B (Weights & Biases) is disabled.
* To enable **TensorBoard**, modify `training_args`:

  ```python
  from transformers import TrainingArguments

  training_args = TrainingArguments(
      output_dir="./results",
      evaluation_strategy="epoch",
      logging_dir="./logs",       # enable tensorboard logs
      report_to=["tensorboard"]   # log to tensorboard
  )
  ```

Run TensorBoard:

```bash
tensorboard --logdir ./logs
```

---

## 📂 Project Structure

```
.
├── fine_tune.ipynb   # Jupyter notebook with training pipeline
├── README.md         # Project documentation
```

---

## 🛠️ Customization

* Replace `tokenized_dataset` with your own dataset
* Adjust `compute_metrics` to use accuracy, F1, etc.
* Modify `training_args` (batch size, learning rate, epochs) for your task

---

## ✅ Example Usage

Run all cells in **fine\_tune.ipynb** to:

1. Load and tokenize your dataset
2. Initialize `Trainer`
3. Train and evaluate the model
4. Save fine-tuned weights in `./results`

---

Would you like me to **extract the exact `training_args` from your notebook** and include them in the README, so it matches your configuration exactly?
