#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script che allena in sequenza più modelli (Mistral, CodeGemma, CodeLlama)
usando quantizzazione a 4 bit e LoRA adapters su un dataset .csv.

Esempio d'uso:
    python sequential_finetuning.py \\
        --model_choices mistral codegemma codellama \\
        --csv_path "datasets/new_dataset.csv"        \\
        --base_output_dir "./results/"

La funzione main allena i modelli sequenzialmente, eliminando i modelli
dalla memoria dopo l'allenamento di ciascuno.
"""

import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------------------------------------------------------
# Individua moduli target per LoRA su strati 4-bit
# -----------------------------------------------------------------------------
def find_target_modules(model):
    unique_layers = set()
    for name, module in model.named_modules():
        if "Linear4bit" in str(type(module)):
            layer_type = name.split('.')[-1]
            unique_layers.add(layer_type)
    return list(unique_layers)

# -----------------------------------------------------------------------------
# Caricamento di un singolo modello e tokenizer
# -----------------------------------------------------------------------------
def load_model_and_tokenizer(model_choice, max_memory=None):
    """
    model_choice: str in ['mistral', 'codegemma', 'codellama']
    max_memory  : dict con eventuali limiti di memoria GPU/CPU
    """
    if model_choice.lower() == 'mistral':
        base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_choice.lower() == 'codegemma':
        base_model = "google/codegemma-7b"
    elif model_choice.lower() == 'codellama':
        # Sostituisci con la variante CodeLlama desiderata
        base_model = "codellama/CodeLlama-7b-hf"
    elif model_choice.lower() == 'deepseek':
        base_model = "deepseek-ai/deepseek-coder-6.7b-base"
    else:
        raise ValueError("model_choice deve essere in ['mistral', 'codegemma', 'codellama', 'deepseek']")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    if max_memory is None:
        max_memory = {
            0:  "8GiB",
            "cpu": "16GiB"
        }

    print(f"[INFO] Caricamento del modello base '{base_model}' (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory
    )

    print("[INFO] Caricamento del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[INFO] Preparazione del modello per k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Configurazione LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_target_modules(model),
    )
    print("[INFO] Applicazione di LoRA al modello...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

# -----------------------------------------------------------------------------
# Funzione di tokenizzazione
# -----------------------------------------------------------------------------
def tokenize_function(examples, tokenizer, max_length=256):
    separator = "\n###\n"
    permission_df = examples["permission_df"]
    filter_code   = examples["filter_code"]

    full_text = [desc + separator + code for desc, code in zip(permission_df, filter_code)]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    # Tokenizzo il prompt da solo
    prompt_text = [desc + separator for desc in permission_df]
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    prompt_lengths = [
        sum(p_id != tokenizer.pad_token_id for p_id in p_ids)
        for p_ids in tokenized_prompt["input_ids"]
    ]

    # Maschero il prompt con -100
    labels = []
    for i, seq in enumerate(tokenized["input_ids"]):
        prompt_len = prompt_lengths[i]
        masked_labels = [-100]*prompt_len + seq[prompt_len:]
        labels.append(masked_labels)

    tokenized["labels"] = labels
    return tokenized

# -----------------------------------------------------------------------------
# Funzione per metriche di valutazione
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred, tokenizer):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if predictions.dtype not in [np.int32, np.int64]:
        predictions = predictions.argmax(axis=-1)

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    for pred, ref in zip(decoded_preds, decoded_labels):
        bleu_scores.append(sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing))
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

    # METEOR
    meteor_scores = [
        meteor_score([ref.split()], pred.split())
        for pred, ref in zip(decoded_preds, decoded_labels)
    ]
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0

    # ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred) for pred, ref in zip(decoded_preds, decoded_labels)]

    avg_rouge1 = np.mean([s["rouge1"].fmeasure for s in rouge_scores]) if rouge_scores else 0.0
    avg_rouge2 = np.mean([s["rouge2"].fmeasure for s in rouge_scores]) if rouge_scores else 0.0
    avg_rougeL = np.mean([s["rougeL"].fmeasure for s in rouge_scores]) if rouge_scores else 0.0

    return {
        "bleu": avg_bleu,
        "meteor": avg_meteor,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
    }

# -----------------------------------------------------------------------------
# Funzione principale: allena più modelli in sequenza e rilascia la memoria
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sequential fine-tuning di più LLM in 4-bit con LoRA.")
    parser.add_argument("--model_choices", nargs='+', default=[ "codegemma", 'deepseek', "codellama","mistral",],
                        help="Lista dei modelli da allenare in sequenza. Es: mistral codegemma codellama")
    parser.add_argument("--csv_path", type=str, default="datasets/new_dataset.csv",
                        help="Path al CSV che contiene le colonne 'permission_df' e 'filter_code'.")
    parser.add_argument("--base_output_dir", type=str, default="./results/",
                        help="Directory base dove salvare i modelli addestrati.")
    args = parser.parse_args()

    # Carico il dataset
    df = pd.read_csv(args.csv_path)
    df.dropna(subset=["permission_df", "filter_code"], inplace=True)
    #df.drop_duplicates(subset=["permission_df", "filter_code"], inplace=True)
    print(f"there are {len(df)} rows in the dataframe.")
    df.reset_index(drop=True, inplace=True)

    # Split train/val
    train_df, eval_df = train_test_split(df, test_size=0.356, random_state=42)
    print(f"there are {len(eval_df)} rows in the test dataframe.")
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset  = Dataset.from_pandas(eval_df)

    # Prepara la tokenizzazione
    def wrapped_tokenize(examples):
        return tokenize_function(examples, tokenizer, max_length=256)

    # Ciclo su ciascun modello
    for model_choice in args.model_choices:
        print("\n" + "="*70)
        print(f"[INFO] Inizio addestramento per il modello: {model_choice}")
        print("="*70)

        # Carico e preparo modello + tokenizer
        model, tokenizer = load_model_and_tokenizer(model_choice)

        # Tokenizzo dataset (ogni volta ricreo i token perché usiamo tokenizer differente)
        print("[INFO] Tokenizzazione dataset per:", model_choice)
        tokenized_train = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer, max_length=256),
            batched=True
        )
        tokenized_eval = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, max_length=256),
            batched=True
        )

        # Imposta parametri di training
        output_dir_model = os.path.join(args.base_output_dir, f"best_model_{model_choice}")
        training_args = TrainingArguments(
            output_dir=f"{output_dir_model}\\{model_choice}",
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            learning_rate=1e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=False,
            fp16=False,
            bf16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metrics(x, tokenizer),
        )

        print(f"[INFO] Avvio training del modello '{model_choice}'...")
        trainer.train()

        # Salvo i pesi finali
        print(f"[INFO] Salvataggio del modello '{model_choice}' in {output_dir_model}")
        trainer.save_model(output_dir_model)

        # Rimuovo il modello dalla memoria
        print(f"[INFO] Rimozione del modello '{model_choice}' dalla memoria...")
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[DONE] Tutti i modelli allenati in sequenza e rimossi dalla memoria.")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
