# QKLSTM data: Air Quality Data in India (2015 - 2020), `city_day.csv`

- File: `city_day.csv` - daily air-quality measurements per Indian city, 2015-01-01 .. 2020-07-01;
  29,532 lines (one header + 29,531 rows), 2,574,056 bytes. Columns: City, Date, PM2.5, PM10, NO, NO2, NOx,
  NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket.
- sha256: `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847`
- Source: Kaggle dataset "Air Quality Data in India (2015 - 2020)", `rohanrao/air-quality-data-in-india`,
  <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>, version 12 (last updated 2020-07-28).
  The dataset states its data was made publicly available by the Central Pollution Control Board (CPCB,
  <https://cpcb.nic.in/>), the source the paper cites ([31]).
- Licence: CC0: Public Domain (as stated on the Kaggle page).
- Obtained: downloaded from the Kaggle page by the repository maintainer (2026-11-11); only `city_day.csv` of
  the dataset's five files is kept (the others are large and not used). The file is unmodified.
- Caveat: a community mirror of CPCB data; the paper's own extract (Bengaluru, 2015-2020) cannot be checked
  against it byte for byte.
