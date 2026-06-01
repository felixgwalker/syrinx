# Syrinx

**An acoustic phylogenetics pipeline.** Syrinx treats birdsong as sequence data — segmenting recordings into syllables, clustering them into a discrete vocabulary, and using alignment-based distances to reconstruct evolutionary relationships between species. The pipeline tests whether acoustic similarity recovers known molecular phylogenies, and provides tools for visualising vocabularies, distance matrices, and tanglegrams.

The name comes from the *syrinx*, the vocal organ in birds that produces all birdsong.

> **Status:** Research prototype accompanying a preprint (in preparation). Results are exploratory and should be interpreted alongside field data and expert judgement.

---

## What it does

1. **Ingests** birdsong recordings (Xeno-canto, Macaulay Library, or local files).
2. **Segments** recordings into syllables using BirdNET or an energy-based fallback.
3. **Clusters** syllables into a discrete vocabulary (letters A, B, C…) via UMAP + HDBSCAN on learned acoustic embeddings.
4. **Aligns** per-species syllable strings and computes pairwise acoustic distances.
5. **Compares** the resulting acoustic tree against a molecular reference tree (Mantel tests, Robinson-Foulds distance, tanglegrams).
6. **Visualises** vocabularies, UMAP embeddings, complexity distributions, and trees.

---

## Repository layout

```
Syrinx/
├── syrinx/           # Pipeline modules
├── notebooks/        # Exploratory analysis (run pipeline first)
├── web/              # Web app — results viewer and public-facing pages
├── docs/             # Paper, methods notes, figure sources
├── tests/            # Unit and integration tests
├── run_pipeline.py   # Pipeline entry point
├── config.yaml       # Pipeline configuration
└── requirements.txt
```

The pipeline and web app are independent — the pipeline runs standalone, and the web app reads its outputs.

---

## Quick start (CLI)

```bash
git clone https://github.com/Felix-G-Walker/Syrinx.git
cd Syrinx
pip install -r requirements.txt
python run_pipeline.py --config config.yaml
```

Outputs land in `results/` — distance matrices, trees (Newick), figures (PNG + interactive HTML), and a JSON summary of the run.

### Requirements

- Python 3.10+
- `ffmpeg` (system install)
- `kaleido` for static figure export
- See `requirements.txt` for Python packages

---

## Web app

The web app provides three interfaces:

- **Researcher** — upload recordings, run the pipeline, view results
- **Field** — population-level vocal-diversity diagnostics with traffic-light status
- **Public** — science-communication piece presenting pre-computed results

Run locally:

```bash
cd web
# follow web/README.md for stack-specific instructions
```

---

## Reference data required before first run

### Zebra finch labels (required — pipeline will abort without this)

The vocabulary validation (§2.5) checks that the HDBSCAN hyperparameters can recover biologically coherent syllable types on a tractable test case — zebra finch (*Taeniopygia guttata*) using published labels from Tchernichovski et al. (2000).

The pipeline automatically downloads zebra finch recordings from Xeno-canto and extracts their features on first run, producing:

```
data/reference/zebrafinch_features.npy   ← auto-generated
data/reference/zebrafinch_labels.npy     ← optional but recommended
```

**`zebrafinch_labels.npy` should be supplied manually.** It should be a NumPy integer array of published syllable-type labels from Tchernichovski et al. (2000), with one label per row of `zebrafinch_features.npy`. The pipeline will print a warning with the required row count after generating the features file.

Without `zebrafinch_labels.npy` the zebra finch macro-F1 gate is skipped (auto-passed) with a warning; the remaining three vocabulary validation gates still run.

### Molecular reference trees (optional — H1 MRM, PGLS, Figure 5 tanglegram)

The H1 multiple regression on distance matrices, the PGLS secondary analyses, and the Figure 5 tanglegram all require a molecular reference phylogeny for *Phylloscopus*. Without these files the pipeline runs to completion but skips those analyses, logging a warning.

Two trees are expected:

| File | Source |
|------|--------|
| `data/trees/alstrom2018.nwk` | Alström et al. (2018) *Phylloscopus* molecular phylogeny — obtain from the paper's Dryad or TreeBASE deposit |
| `data/trees/tietze2015.nwk` | Tietze et al. (2015) *Mol. Phyl. Evol.* — obtain from the paper's supplementary data |

Both files must be in Newick format with tip labels matching the binomial species names used in the pipeline (e.g. `Phylloscopus_trochilus`). Alternatively, pre-compute the patristic distance matrix and save it to `data/distances/molecular_distances.pkl` (a NumPy array or `{"D": array}` dict); the pipeline loads that first and does not need the `.nwk` files.

**TODO:** obtain these tree files from their published deposits and add them to `data/trees/`.

### AVONET traits (optional — Figure 6b habitat annotation)

The four-corner convergent evolution diagnostic (Figure 6b) annotates species pairs with their habitat type using AVONET (Tobias et al. 2022). Without this file the figure is still generated, but hover text shows plain pair labels rather than habitat concordance information.

To enable annotation:

1. Download **AVONET1_BirdLife.csv** from the AVONET Figshare dataset (Tobias et al. 2022, *Ecology Letters*).
2. Place it at `data/avonet_traits.csv`.

The loader expects columns `Species1` and `Habitat`; the standard AVONET1_BirdLife.csv file provides both without modification.

---

## Reproducing the paper

The full set of results in the accompanying preprint can be regenerated from a clean checkout:

```bash
python run_pipeline.py --config config.paper.yaml
```

This will pull the species list, fetch recordings, run segmentation and clustering, compute distance matrices, run the Mantel and Robinson-Foulds tests, and render all figures. Expect a multi-hour run depending on hardware and network.

---

## Citing

If you use Syrinx in published work, please cite the preprint (citation to follow on bioRxiv deposit) and link to this repository.

---

## Author

**Felix Walker** — independent researcher, Edinburgh, Scotland.

---

## License

MIT — see [LICENSE](LICENSE).
