# Company DNA — Semantic Classification of Public Companies

Sector labels put **Meta, Alphabet, and Verizon in the same bucket**. Intuitively these
are not the same business, and treating them as peers distorts any comparison, peer
group, or model built on that grouping.

This project tests whether company *descriptions* carry enough signal to build a
data-driven representation of what a company actually does — its **"DNA"** — and whether
that representation groups companies more sensibly than sectors do.

It produces three things:

| Output | What it is |
|---|---|
| **Semantic search** | Find companies by what they do, in plain English |
| **Semantic factors** | 16 continuous coordinates locating each company in thematic space |
| **Custom clusters** | Peer groups derived from description similarity, not a fixed taxonomy |

---

## Repository layout

```
dataCollection.ipynb   Pulls company data from the production MySQL DB -> Parquet
notebook.ipynb         The research workflow: embeddings -> factors -> clusters -> analysis
embed.py               Standalone embedding job (slow step, cached to disk)
RESEARCH_REPORT.md     Findings write-up
data/
  companyStatesFull.parquet   Source: ticker, name, description, exchange, sector, industry
  companyDNA.parquet          Output: source columns + dna_01..16 + cluster + map coords
  embeddings/                 Cached embedding matrix (gitignored, regenerable)
figures/                      Charts produced by the notebook
```

---

## Setup

Dependencies (already present in the `python310` conda env used here):

```bash
pip install sentence-transformers umap-learn hdbscan pyarrow python-dotenv
```

Database credentials for `dataCollection.ipynb` come from a local `.env` file
(gitignored — copy `.env.example` and fill it in):

```
DB_HOST=your-rds-endpoint
DB_USER=...
DB_PASSWORD=...
DB_NAME=...
```

> **Note on `USE_TF=0`.** This environment has Keras 3 installed, which the `transformers`
> TensorFlow backend refuses to import. Both `embed.py` and the notebook set
> `os.environ["USE_TF"] = "0"` before any HuggingFace import so only the PyTorch path
> loads. Without it, `import sentence_transformers` raises a `RuntimeError`.

---

## Running it

The embedding step is the only slow part (~10 min on Apple Silicon MPS). It is cached, so
it runs once:

```bash
python embed.py                 # writes data/embeddings/bge_base_v1p5.npy
jupyter lab notebook.ipynb      # everything else runs in seconds
```

`notebook.ipynb` regenerates the embeddings automatically if the cache is missing, so
running the notebook alone also works.

---

## Method

**Embeddings.** `BAAI/bge-base-en-v1.5` (768-dim, 512-token context), run locally on MPS.
Chosen over the more common `all-MiniLM-L6-v2` because MiniLM's 256-token limit would
truncate **33%** of these descriptions; BGE truncates **0.7%**. Vectors are L2-normalised,
so cosine similarity is a dot product.

**Factors.** PCA to 16 components, then re-normalised to unit length. This is deliberate:
the factors are **direction cosines**, capturing *where a company points* in thematic
space rather than how much of a theme it has. A longer description cannot inflate a score.

**Clusters.** Spherical k-means (k-means on unit-norm vectors) in a 50-dimensional PCA
space. Silhouette declines monotonically with `k` — the semantic space is a continuum, not
a set of islands — so `k` is chosen against a stated design criterion (must be finer than
the 11 sectors it aims to improve on) rather than a naive argmax.

**Labels.** Class-based TF-IDF over pooled cluster descriptions produces the distinctive
vocabulary for each group.

---

## Key results

See [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for the full write-up. In brief:

- **The embedding recovers sub-industry structure sectors flatten.** Verizon's nearest
  neighbours are AT&T, Rogers, and T-Mobile; Meta's are Pinterest, Snap, and Weibo —
  including one peer the taxonomy files under a different sector entirely.
- **Communication Services is not one neighbourhood but several.** Its strongest internal
  bonds are carrier↔carrier and entertainment↔entertainment, not the sector as a whole.
- **The clusters find groupings sectors cannot express** — including a China-based online
  platform cluster that the sector taxonomy scatters across four different sectors.
- **It fails on boilerplate.** Blank-check/SPAC shells cluster *more tightly* than real
  businesses because their filings share template language. Filter them before clustering.
- **It fails on conglomerates.** Berkshire, GE, and Danaher sit far from every centroid;
  their continuous factor vector is more honest than any single cluster label.

---

## ⚠️ Look-ahead warning

Every description describes the company **as it is today**. Using today's Nvidia
description to characterise Nvidia in 2015 injects knowledge of the AI buildout into a
period when it had not happened.

**These features are safe for cross-sectional work today and unsafe for historical
backtests.** Fixing this requires point-in-time descriptions — 10-K Item 1 text, filed
annually and timestamped — embedded per year to build a company-year panel. That is the
precondition for any predictive use and is not done here.

---

## Scope

This is a representation-building exercise, not a trading strategy. No returns data is
used and nothing is backtested.
