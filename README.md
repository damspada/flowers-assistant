# Flowers Assistant — IG.3513 Advanced Deep Learning project

Team 10 (TP Group 2) · Dataset: [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

A multimodal assistant that combines an ML classifier, a DL classifier, an image generator and an LLM with RAG, integrated with LangChain in a Gradio app.

## Roles

| Role | Component | Who |
|---|---|---|
| 1 | Data pipeline, EDA, ML baseline, coordination | Damiano Spadaccini |
| 2 | Deep learning classifier | |
| 3 | Generative model | |
| 4 | RAG (Ollama) | |
| 5 | LangChain + Gradio integration | |

## Setup

```bash
git clone <repo-url>
cd flowers-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import flowers_data; flowers_data.download()"   # ~350 MB, only once
```

On Colab: clone the repo, `%cd flowers-assistant`, then `pip install -r requirements.txt` (torch is already installed).

## Data pipeline (`flowers_data.py`)

We use the **official split**: train 1,020 images (10 per class), val 1,020 (10 per class), test 6,149. Hyperparameters are chosen on `val`; `test` is used only once, for the final results.

```python
from flowers_data import get_split, load_image, to_tensor, CLASS_NAMES, set_seed

set_seed()                              # SEED = 42
paths, labels = get_split("train")      # list of image paths, labels in 0..101
img = load_image(paths[0])              # PIL image, RGB
x = to_tensor(img)                      # (1, 3, 224, 224), ImageNet normalisation
print(CLASS_NAMES[labels[0]])
```

`preprocess` / `to_tensor` must be used for **every** classifier input (dataset images, uploaded images and generated images). Training augmentation goes on top of it.

## Interfaces between components

| Function | Input | Output | Role |
|---|---|---|---|
| `classify_ml(image)` | PIL image | `{"class_id", "label", "confidence", "probs"}` | 1 |
| `classify_dl(image)` | PIL image | `{"class_id", "label", "confidence", "probs"}` | 2 |
| `generate(prompt, ref_image=None)` | text (+ optional PIL image) | PIL image | 3 |
| `ask_rag(question, predicted_class=None)` | text (+ optional class name) | `{"answer", "sources"}` | 4 |

`probs` is a list of 102 probabilities in the order of `CLASS_NAMES`.

## Structure

```
flowers_data.py     shared data pipeline (splits, class names, preprocessing)
notebooks/          EDA, baseline and experiments
requirements.txt
```
