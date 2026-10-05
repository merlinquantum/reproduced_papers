# Data acquisition — Q-CHIPP

**Overall status: READY for the selected binding and immunogenicity simulator experiments.** The original supplement files were verified on 2026-10-02. Clinical and hardware assets remain outside the selected scope.

## Acquisition

DataS1 and the supplement PDF are distributed by Europe PMC as a single zip archive:

```text
https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13398535/supplementaryFiles
```

`lib/data.py` downloads that archive on first use into `data/QCHIPP/paper_supplement/` and verifies the SHA-256 digest of the extracted CSV before any experiment uses it.

Related source links:

- Source article: https://pmc.ncbi.nlm.nih.gov/articles/PMC13398535/
- Article DOI: 10.1126/sciadv.aec3824

## Verified files

| File | Bytes | SHA256 | Verification |
|---|---:|---|---|
| sciadv.aec3824_data_file_s1.zip | 315460 | 892c17bdb0f18da113f4515ec65ee29533e8033cd0bb2c69f0886090e85b465f | PK magic, CRC test passed, sole member aec3824_data_file_s1.csv |
| sciadv.aec3824_sm.pdf | 2083013 | 158bdcca4031fd2a69c6df61770b5c71a8868bcd3874bcf66e67d9e42b95829e | PDF 1.6, 8 pages, unencrypted, exact paper title/DOI, all text extracted and read |
| aec3824_data_file_s1.csv | 1577896 | 2b91dde1046d46f9ec0bd64ae1d2bc0d148da09b065287aef44a1238c0bb3b07 | Exact named member extracted without arbitrary archive paths; 8470 rows / 24 columns |

A dataset-specific licence is not separately stated in the archive or the PDF; retain the source paper CC-BY-NC-4.0 attribution and do not infer unrestricted licensing.

## Dataset facts

All 8470 peptides are unique and HLA-A02:01. Binding labels: 2883 Binder / 5587 NonBinder. Immunogenicity: 1037 Immunogenic / 7433 Non_Immunogenic. Two noncanonical-X sequences exist but neither occurs in any released training or test split. One row `DLPLIIQRL` has both MHC.Assay and T.cell.response = NA despite a derived Binding=Binder annotation; it is in the binding test split. The conservative main analysis excludes this ambiguous assay-truth row, leaving 999 binding test peptides. Do not silently replace missing assay truth with predicted affinity or the derived label; a source-annotation sensitivity evaluation may be separate.

Released split columns: Binding_split 150 train (75/75) and 1000 test (500/500); Immunogenic_mix_split 300 train (150/150) and 1000 test (500/500); Immunogenic_split 300 train (150/150) and 1000 test (500/500). The binder-only split contains 1300 experimentally annotated binders. There is no train/test peptide overlap in any task. Actual test sizes differ from the main-paper 50/100 descriptions; preserve release splits and label comparisons as release-split or partial, not exact numerical table reproduction. No official validation split or checkpoints are included; initial fixed-budget experiments avoid tuning or selection on test data.

Training and test core fields: Mutant_peptide, Binding, Immunogenicity, MHC.Assay, T.cell.response, HLA_allele, MT.BA_Rank and the three released split columns. BA is an optional predicted feature (`MT.BA_Rank < 2`), never binding truth. The supplement legend explicitly identifies experimental binding assays and functional T-cell assays as the truth sources.

Dataset contents cannot be inferred from the ZIP filename. Verification checks PDF/ZIP magic, complete archive integrity, safe extraction paths (no traversal), visible licences, actual tables, sheets, schema and row counts, HLA allele, canonical nine-mer residues, experimental binding labels (not BA prediction), immunogenicity labels, exact split and checkpoint assets, and peptide overlap or conflicting labels. Every acquired file is then checksummed and its release provenance recorded.

## Public author assets inspected, not used as a fallback

Original author repository release v1.0.0, commit `9f72bd9fc87522bc45c806f9849d07210a64e48d` (Apache-2.0 repository licence), source https://github.com/IBM/Q-CHIPP/tree/v1.0.0. The current main-branch tree was also inspected via the GitHub API and exposes the same NeoantigenData family, not a verified IEDB 8470 dataset.

`input_data/NeoantigenData/mutant_peptide_df_categorical.csv`:
- Observed 775 rows, 24 allele IDs, 750 unique nine-position peptide vectors.
- Tcell_response 742 negative / 33 positive; HLA-A02:01 maps to hla9 with 290 rows.
- Columns: serialized index, motif0–motif8 (amino-acid integer codes), hla, Tcell_response, ba_rank.
- **No experimental MHC-binding label exists in this file.** `ba_rank` is a predicted affinity rank and must not be used as experimental binding truth or to claim the paper's binder-only training population.
- Source byte size 29572; SHA256 `b95a05507c16dba8f7ee481ea07f575fb4be7c10c2b104bfd83b86b11ab7ce5f`.
- Released split 1 contains 620 training rows (24 positive) and 155 test rows (9 positive). These are **not** the paper's stated 150/50 binding or 300/100 binder-only immunogenicity splits. Duplicate exact sequence identities across alleles are possible and must be group-audited before any justified substitute study.
- Released amino_acid_map.csv provides canonical alphabetical A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y codes 0–19; HLA_map.csv lists 24 alleles.

Extended batcell CSVs contain physicochemical sequence features, Ba_rank_cate and Ba_Tcell but do not establish an experimental binding label. Never relabel a predicted binder as an experimentally confirmed binder.

The author figure repository `qchipp_publication`, commit `4b19620deb7e890bd2a306c4bc518879d6e2b2db`, contains notebooks, figures and code but no standalone CSV/TSV/NPY/NPZ/XLSX/RDS/RData data assets were found in the initial inventory. Some notebooks may embed outputs; those are source artifacts, not newly obtained training results.

These older 775-row author assets are not used by this reproduction.
