# Data Sources

## Required files

Download the paper's curated inputs directly from the
[authors' pinned data directory](https://github.com/orcacomputing/pep-q-gan/tree/51f7dfa162e33a3be9ee90a69dd93f92e07d7788/data).

| File | Purpose |
|---|---|
| `HLA_pos9mers.txt` | Positive peptide–HLA training pairs |
| `MHC_pseudo_HLA_only.dat` | HLA conditioning pseudosequences provided in the authors' repository |
| `unique_hla_allotypes.txt` | 131-HLA evaluation panel |

Preserve the filenames in external `data/pep_q_gan/`, or specify an exact directory
with `--dataset-root`. Pinned URLs, sizes and hashes are in
[training_data_sources.json](assets/training_data_sources.json). The acquisition
command downloads those files and checks their hashes:

```bash
python implementation.py --paper pep_q_gan --stage prepare-data
```

The training file has 105,971 rows across 126 HLA, including two peptides with `X`.
The full model preserves them and their order. The paper reports 105,970 rows.
The notebook's optional small author-data exercise applies its own documented
canonical-9-mer filter and does not replace the full training dataset.

## Original sources and use conditions

The paper describes the peptide file as curated
[IEDB](https://www.iedb.org/) ligands. A current IEDB export
did not reproduce the supplied historical dataset, so the author files remain
the reproduction inputs. Cite the peptide-generation paper, its repository,
IEDB and the [NetMHCpan-4.1 dataset paper](https://doi.org/10.1093/nar/gkaa379).

The HLA pseudosequences in `MHC_pseudo_HLA_only.dat` are provided in the
[authors' repository](https://github.com/orcacomputing/pep-q-gan/blob/51f7dfa162e33a3be9ee90a69dd93f92e07d7788/data/MHC_pseudo_HLA_only.dat)
as a separate conditioning input. Their original source and filtering
procedure have not been independently verified. 
See [IEDB terms](https://www.iedb.org/terms_of_use_v3.php) and
[IPD-IMGT/HLA conditions](https://www.ebi.ac.uk/ipd/imgt/hla/licence/) where applicable.

Users obtain source files directly; raw datasets are not redistributed here.
MHCflurry is installed through pip and its weights through the official downloader. 
Follow upstream attribution and use conditions for independently acquired data and models.
