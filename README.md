# Small Language Model from Scratch

An educational implementation of a small decoder-only language model built with PyTorch. The project covers the complete workflow: downloading text data, training a byte-level BPE tokenizer, preparing language-modeling batches, building and training a causal Transformer, generating text, and supervised fine-tuning on Bhagavad Gita conversations.

## What is included

- Streaming download of a 100 MB subset of the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
- A byte-level BPE tokenizer with a vocabulary size of 1,000
- Token and positional embeddings
- Multi-head causal self-attention and feed-forward Transformer blocks
- Training and validation loops with checkpointing and perplexity evaluation
- Top-k and temperature-based text generation
- Supervised fine-tuning on a Bhagavad Gita conversational dataset
- Saved tokenizer, pretrained model, and fine-tuned model artifacts

## Project structure

```text
slm-from-scratch/
├── data/
│   ├── data_download.py                  # TinyStories streaming downloader
│   ├── bhagwat_geeta_data_download.ipynb # SFT data download and cleaning
│   └── bhagavad_gita_sft_*.json          # Fine-tuning datasets
├── model/
│   ├── data_prep.ipynb                   # End-to-end model development
│   ├── Embadding.ipynb                   # Embedding/model experiments
│   ├── Fast_traning.ipynb                # Optimized training workflow
│   ├── fine_tuning.ipynb                 # Supervised fine-tuning workflow
│   ├── saved_slm/                        # Pretrained model artifacts
│   └── fine_tuned_slm/                   # Fine-tuned checkpoints
├── tokenizer/
│   ├── token.ipynb                       # BPE tokenizer training
│   └── tokenizer/tokenizer.json          # Saved tokenizer
├── evaluation/                           # Reserved for evaluation code
├── inference/                            # Reserved for inference code
├── training/                             # Reserved for training code
├── config.py
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/nikunjbharti/SLM-from-scrach.git
cd SLM-from-scrach
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To work with the notebooks, also install and launch Jupyter:

```bash
pip install jupyter
jupyter notebook
```

## Usage

Run commands from the repository root unless a notebook notes otherwise.

### Download the pretraining data

```bash
python data/data_download.py
```

This streams TinyStories examples into `tinystories_100mb.jsonl` until the file reaches approximately 100 MB. The generated file is intentionally excluded from Git.

### Train the tokenizer

Open `tokenizer/token.ipynb` and run its cells in order. It trains a byte-level BPE tokenizer and saves the resulting tokenizer configuration under `tokenizer/tokenizer/`.

### Train the base model

Open `model/Fast_traning.ipynb` for the main training workflow. The notebook:

1. Loads the saved tokenizer and TinyStories data.
2. Encodes the corpus and creates next-token prediction samples.
3. Builds a decoder-only causal Transformer.
4. Trains and validates the model.
5. Saves checkpoints and generates sample text.

`model/data_prep.ipynb` contains a more step-by-step version of the same development process.

### Fine-tune the model

1. Use `data/bhagwat_geeta_data_download.ipynb` to download and clean the conversational dataset.
2. Open `model/fine_tuning.ipynb` and run its cells in order.
3. The notebook loads `model/saved_slm/model_weights.pt`, prepares supervised examples, trains the model, and saves the best weights in `model/fine_tuned_slm/`.

## Default model configuration

The current notebooks use these main settings:

| Parameter | Value |
| --- | ---: |
| Vocabulary size | 1,000 |
| Context length | 512 tokens |
| Embedding dimension | 96 |
| Attention heads | 6 |
| Transformer layers | 4 |

The notebooks automatically select CUDA when it is available and otherwise fall back to CPU. Training on CPU can take considerably longer.

## Notes

- This repository is notebook-driven and intended for learning and experimentation.
- Some notebooks contain fallback paths so they can be launched from either the repository root or their own directory. If a file is not found, verify the notebook's working directory first.
- Model weights can be large. Avoid committing additional checkpoints unless they are intentionally meant to be versioned.
- The repository name uses the existing spelling `SLM-from-scrach` in its GitHub URL.

## Acknowledgements

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for the pretraining corpus
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) and Tokenizers for data access and tokenization utilities
- [PyTorch](https://pytorch.org/) for model implementation and training
