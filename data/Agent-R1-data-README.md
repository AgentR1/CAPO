# Agent-R1 Data

Preprocessed datasets and runtime artifacts for reproducing the agentic reinforcement learning experiments in [**Agent-R1**](https://github.com/AgentR1/Agent-R1): *Training Powerful LLM Agents with End-to-End Reinforcement Learning*.

These files cover the main benchmarks used in the Agent-R1 paper and codebase: **ALFWorld**, **WebShop (full)**, **HotpotQA**, **Paper Search (PaSa)**, and cross-corpus retrieval corpora (**HotpotQA**, **2WikiMultiHopQA**, **MuSiQue**).

**Total size:** ~19 GB (compressed where noted).

## Repository Layout

```
Agent-R1-data/
├── alfworld/                 # ALFWorld games + ARFT parquet splits
├── webshop_full/             # WebShop full-mode runtime artifacts
├── hotpotqa/                 # HotpotQA train/validation parquet
├── hotpotqa_corpus/          # HotpotQA retrieval corpus + FAISS index
├── musique_corpus/           # MuSiQue retrieval corpus + FAISS index
├── 2wikimultihopqa_corpus/   # 2Wiki retrieval corpus (jsonl only)
└── pasa/                     # Paper Search (PaSa) train/test parquet
```

## Dataset Overview

| Subdirectory | Benchmark | Size | Description |
|---|---|---:|---|
| `alfworld/` | ALFWorld | ~1.3 GB | TextWorld household tasks. `games.zip` contains 11,481 `game.tw-pddl` files; parquet splits for train / valid_seen / valid_unseen. |
| `webshop_full/` | WebShop (full) | ~9.7 GB | Full shopping environment: SQLite product store (~1.18M products), Lucene/Pyserini index, goals, train/test parquet. |
| `hotpotqa/` | HotpotQA | ~29 MB | Training and evaluation parquet for multi-hop QA. |
| `hotpotqa_corpus/` | HotpotQA retrieval | ~4.2 GB | `hpqa_corpus.jsonl` (509,308 passages), BGE embeddings (`hpqa_corpus.npy`), FAISS index (`index.bin`). |
| `musique_corpus/` | MuSiQue retrieval | ~1.2 GB | `hpqa_corpus.jsonl` (139,416 passages), embeddings, and FAISS index. |
| `2wikimultihopqa_corpus/` | 2WikiMultiHopQA retrieval | ~2.9 GB | `hpqa_corpus.jsonl` only (5,902,082 passages). **Index not included** — see [Rebuilding 2Wiki Index](#rebuilding-2wiki-index) below. |
| `pasa/` | Paper Search | ~2 MB | PaSa-style train/test parquet for academic paper discovery. |

### Split Statistics

**ALFWorld** (`alfworld/`)

| Split | Rows |
|---|---:|
| train | 3,553 |
| valid_seen | 140 |
| valid_unseen | 134 |

After download, unzip `alfworld/games.zip` to restore the `games/` directory expected by the training scripts.

**WebShop full** (`webshop_full/`)

| Item | Count |
|---|---:|
| Products | 1,181,430 |
| Goals | 12,087 |
| Train rows | 11,587 |
| Test rows | 500 |

Split convention: `test = goals[:500]`, `train = goals[500:]`.

**HotpotQA** (`hotpotqa/`)

| Split | Rows |
|---|---:|
| train | 90,447 |
| validation | 7,405 |

Also includes cross-corpus validation parquet:

- `2wikimultihopqa_validation.parquet`
- `musique_validation.parquet`

**Paper Search** (`pasa/`)

| Split | Rows |
|---|---:|
| train | 33,551 |
| test | 50 |

## Download

### ModelScope (recommended)

```bash
pip install -U modelscope

# Login (get token from https://www.modelscope.cn/my/myaccesstoken)
modelscope login --token "$MODELSCOPE_TOKEN"

# Download everything
modelscope download \
  --dataset Melmaphother/Agent-R1-data \
  --local_dir ./data
```

Download a single subdirectory:

```bash
modelscope download \
  --dataset Melmaphother/Agent-R1-data \
  --local_dir ./data/webshop_full \
  webshop_full
```

### Post-download Setup

**ALFWorld** — unpack game files:

```bash
cd data/alfworld
unzip games.zip
```

**WebShop** — point the environment server to the artifacts:

```bash
export WEBSHOP_DATASET_MODE=full
export WEBSHOP_INDEX_DIR=/path/to/data/webshop_full
```

**HotpotQA retrieval** — set the corpus root for search tools:

```bash
export HOTPOTQA_CORPUS_DATA_ROOT=/path/to/data/hotpotqa_corpus
```

## Rebuilding 2Wiki Index

The `2wikimultihopqa_corpus/` directory ships with `hpqa_corpus.jsonl` only. To rebuild `index.bin` and `hpqa_corpus.npy` (requires GPU + BGE-large-en-v1.5):

```bash
python recipe/hotpotqa/process_hotpotqa.py \
  --data_dir data/corpus/2wikimultihopqa_corpus \
  --corpus_path data/corpus/2wikimultihopqa_corpus/hpqa_corpus.jsonl \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --batch_size 1024
```

Expected output size: ~45 GB (`hpqa_corpus.npy` + `index.bin`). Only `index.bin` is required at inference time; `hpqa_corpus.npy` is an embedding cache.

## Usage with Agent-R1

Clone the [Agent-R1 repository](https://github.com/AgentR1/Agent-R1) and place the downloaded data under `data/` (or override paths via environment variables in the training scripts under `examples/` and `recipe/`).

Typical paths referenced by the codebase:

| Task | Data paths |
|---|---|
| ALFWorld | `data/alfworld/train.parquet`, `data/alfworld/games/` |
| WebShop | `data/webshop_full/train.parquet`, `data/webshop_full/` (env) |
| HotpotQA | `data/corpus/hotpotqa/train.parquet`, `data/corpus/hotpotqa_corpus/` |
| Paper Search | `data/pasa/train.parquet` |

## License

Please refer to the original benchmark licenses:

- [ALFWorld](https://github.com/alfworld/alfworld)
- [WebShop](https://github.com/princeton-nlp/WebShop)
- [HotpotQA](https://hotpotqa.github.io/)
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)
- [PaSa / Paper Search](https://github.com/bytedance/pasa)

## Citation

If you use this data with Agent-R1, please cite:

```bibtex
@misc{cheng2025agentr1trainingpowerfulllm,
  title={Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning},
  author={Mingyue Cheng and Jie Ouyang and Shuo Yu and Ruiran Yan and Yucong Luo and Zirui Liu and Daoyu Wang and Qi Liu and Enhong Chen},
  year={2025},
  eprint={2511.14460},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.14460}
}
```
