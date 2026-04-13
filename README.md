# Gpt`s From Zero

gpts from zero is my personal, modular playground for learning how GPT-style models work from first principles. This project is a rebuild and exploration inspired by the excellent [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) book and repository by Sebastian Raschka.

The goal is not just to train models, but to understand every moving piece:

- how text becomes tokens
- how attention creates context
- how blocks become a language model
- how training loops shape behavior

## Why This Repo Exists

Most tutorials show results. This repo is for understanding mechanisms.

You will keep:

- `src/` clean and reusable
- `notebooks/` messy and exploratory
- `docs/` personal notes and "aha" explanations

## Project Structure

```text
gpts-from-zero/
├── docs/
├── data/
├── src/
│   ├── tokenizer.py
│   ├── attention.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── notebooks/
│   ├── 00_pytorch_101.ipynb
│   ├── 01_data_engine.ipynb
│   ├── 02_attention_lab.ipynb
│   └── 03_building_the_brain.ipynb
├── experiments/
├── weights/
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

1. Create and activate environment.
2. Install dependencies.
3. Open notebooks and start experimenting.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Or use:

```bash
./setup.sh
```

## Suggested Learning Path

1. `notebooks/00_pytorch_101.ipynb`
2. `notebooks/01_data_engine.ipynb`
3. `notebooks/02_attention_lab.ipynb`
4. `notebooks/03_building_the_brain.ipynb`
5. Move stable logic from notebooks into `src/`

## Notes

- Keep datasets small while iterating on a Mac.
- Save trained checkpoints into `weights/`.
- Track insights in `docs/` so learning compounds over time.
