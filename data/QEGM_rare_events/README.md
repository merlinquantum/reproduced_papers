# QEGM_rare_events packaged data

`sp500_daily_logreturns_1990_2022.csv` — 8314 daily log-returns of the
S&P 500 index (^GSPC), 1990-01-02 .. 2022-12-30, derived from closing
prices fetched from the public Yahoo Finance chart API. Packaged so the
`papers/QEGM_rare_events` real-data ablation configs
(`sp500_ablations.json`, `sp500_no_tail.json`) run offline from a fresh
clone.

This is *derived* data (log-returns, not prices). Regenerate with:

```bash
python papers/QEGM_rare_events/utils/fetch_sp500.py
```

The date range matches the paper's Sec. VI.D finance description
("daily log-returns of the S&P 500 from 1990 to 2022"). Note the paper
does not specify preprocessing or experiment hyper-parameters, so the
experiments built on this file are an extension testing whether the
reproduction's ablation findings transfer to real heavy-tailed data —
not a reproduction of the paper's finance numbers.
