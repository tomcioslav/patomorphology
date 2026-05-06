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
└── data/
    ├── 1x/
    │   ├── Images/        # 290 TIFFs at native resolution (~11k × 16k px)
    │   └── Masks/         # matching PNG masks, 12-class palette
    ├── 2x/{Images,Masks}/
    ├── 5x/{Images,Masks}/
    ├── 10x/{Images,Masks}/
    └── MarginData/
        ├── Images/        # 290 images for the margin-detection sub-task
        ├── X/             # input arrays
        └── y/             # margin coordinates
```

(`paths.nmsc` in the root `config.py` points at this inner `data/` directory; `paths.nmsc_margin` points at `MarginData/`.)

Class IDs in the masks (12 classes). The order matches the dataset author's
training code (`05_patch_training.py` in their GitHub repo) — that's the
source of truth, not the alphabetical order in the dataset's palette legend.

| ID | Abbrev | Name | RGB |
|---:|---|---|---|
| 0  | EPI | Epidermis | (73, 0, 106) |
| 1  | GLD | Glands | (108, 0, 115) |
| 2  | INF | Inflammation | (145, 1, 122) |
| 3  | RET | Reticular dermis | (181, 9, 130) |
| 4  | FOL | Hair follicles | (216, 47, 148) |
| 5  | PAP | Papillary dermis | (236, 85, 157) |
| 6  | HYP | Hypodermis | (254, 246, 242) |
| 7  | KER | Keratin | (248, 123, 168) |
| 8  | BKG | Background | (0, 0, 0) |
| 9  | BCC | Basal cell carcinoma | (127, 255, 255) |
| 10 | SCC | Squamous cell carcinoma | (127, 255, 142) |
| 11 | IEC | Intraepidermal carcinoma | (255, 127, 127) |

The masks themselves are stored as **RGB-encoded PNGs**, not palette PNGs.
`NMSCDataset._load_mask` decodes the RGB → class index using these colours
(with nearest-colour fallback for anti-aliasing artefacts at boundaries).

## After downloading

Confirm the layout, then we can wire `pato.data` to load image/mask pairs
and tile them for training. Start with the **5× directory** for first
iteration: small enough to fit in memory, large enough to be meaningful.
