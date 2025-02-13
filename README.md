# IFTTT Code Generator

## Project Description
This project compares the performance of three code generation models for IFTTT automation: **GPT-2, BART, and Mistral**. The models have been trained on natural language descriptions to generate the corresponding code.

## Project Structure
The project is organized into the following files and directories:

- **datasets/**: Contains the datasets used for model training and testing.
- **preprocessing_and_cleaning.ipynb**: Notebook for dataset cleaning and preparation.
- **gpt2_nl2ifttt.ipynb**: Notebook for fine-tuning and inference using the GPT-2 model.
- **bart_nl2ifttt.ipynb**: Notebook for fine-tuning and inference using the BART model.
- **fine_tuning_mistral.ipynb**: Notebook for fine-tuning and inference using the Mistral model.
- **test_and_compare_models.ipynb**: Notebook for testing and comparing the models.
- **try_models.ipynb**: Notebook for trying the models on a single prompt.
- **generated_codes.csv**: CSV file containing inference results, including the prompt, generated code from various models, and the reference code.

## Installation
To run the project, make sure you have installed the following dependencies:

```bash
pip install transformers datasets torch pandas scikit-learn peft accelerate bitsandbytes nltk rouge_score evaluate fuzzywuzzy
```

If you are using Google Colab, you may need to mount Google Drive to access the trained models.

## Usage
### 1. Preprocess the Data
Run the preprocessing notebook to clean and prepare the datasets:
```bash
jupyter notebook preprocessing_and_cleaning.ipynb
```

### 2. Train the Models
You can train each model by running the corresponding notebook:
```bash
jupyter notebook gpt2_nl2ifttt.ipynb
jupyter notebook bart_nl2ifttt.ipynb
jupyter notebook fine_tuning_mistral.ipynb
```

### 3. Test and Compare the Models
After training, you can run the comparison notebook to evaluate performance:
```bash
jupyter notebook test_and_compare_models.ipynb
```

### 4. Test the Models on a single prompt
After training, you can run the comparison notebook to evaluate performance:
```bash
jupyter notebook test_and_compare_models.ipynb
```

## Model Evaluation
The models are evaluated using different similarity metrics between the generated code and the reference code:
- **BLEU**: Measures the precision of n-grams compared to the reference.
- **METEOR**: Considers synonyms, stemming, and word order.
- **ROUGE**: Evaluates recall based on overlapping sequences.
- **PERPLEXITY**: Measures model uncertainty.
