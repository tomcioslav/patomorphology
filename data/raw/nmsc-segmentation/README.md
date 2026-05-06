# NMSC Segmentation Dataset (Thomas et al., 2021)

**Non-Melanoma Skin Cancer Segmentation for Histopathology Dataset.**
290 H&E-stained tissue sections (BCC / SCC / IEC) with pixel-level masks
across 12 tissue classes.

- DOI: [10.14264/8be4bd0](https://doi.org/10.14264/8be4bd0)
- Landing page: https://espace.library.uq.edu.au/view/UQ:8be4bd0
- Paper: Thomas et al., *Data in Brief* 39, 107587 (2021)
  https://doi.org/10.1016/j.dib.2021.107587
- License: UQ custom terms — see https://guides.library.uq.edu.au/deposit_your_data/terms_and_conditions

## Why no auto-download

UQ eSpace and the RDM file system are JavaScript SPAs, and the underlying
file API is bot-blocked at CloudFront. There is no public mirror on Kaggle,
Zenodo, or GitHub. Download must be done via a real browser.

## Getting the data

1. Open the landing page in a browser, accept the terms.
2. Download all files via the "Files" / RDM section.
3. Place the archive(s) (and/or the unpacked tree) in **this directory**.
4. The expected unpacked layout is:

```
nmsc-segmentation/
├── 1x/
│   ├── Images/   # 290 TIFFs at native resolution (~11k × 16k px)
│   └── Masks/    # matching PNG masks, 12-class palette
├── 2x/
│   ├── Images/
│   └── Masks/
├── 5x/
│   ├── Images/
│   └── Masks/
├── 10x/
│   ├── Images/
│   ├── Masks/
│   └── margins.csv         # per-image specimen-margin pixel coords
├── 12_class_Palette.tif
└── README.pdf
```

Class IDs in the masks (12 classes):

| ID | Name | Abbrev |
|---:|---|---|
| 0  | Background | BKG |
| 1  | Keratin | KER |
| 2  | Epidermis | EPI |
| 3  | Papillary dermis | PAP |
| 4  | Reticular dermis | RET |
| 5  | Hypodermis | HYP |
| 6  | Glands | GLD |
| 7  | Hair follicles | FOL |
| 8  | Inflammation | INF |
| 9  | Basal cell carcinoma | BCC |
| 10 | Squamous cell carcinoma | SCC |
| 11 | Intraepidermal carcinoma | IEC |

(Verify against `12_class_Palette.tif` and the bundled `README.pdf` after download — palette ID order is the source of truth.)

## After downloading

Confirm the layout, then we can wire `pato.data` to load image/mask pairs
and tile them for training. Start with the **5× directory** for first
iteration: small enough to fit in memory, large enough to be meaningful.
