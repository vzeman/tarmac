# Structural Condition Dataset Catalog

This catalog separates verified research claims from repo-derived integration notes. Rows marked `candidate` are supported by the verified research findings where available. Rows marked `integrated` are already represented in `src/tarmac/datasets/`; their integration details come from the repo code and are labeled as such where not covered by the verified claims.

## Verified Source Base

- SDNET2018: https://ieee-dataport.org/documents/sdnet2018-concrete-crack-image-dataset-machine-learning-applications
- CODEBRIM: https://zenodo.org/records/2620293
- CrackForest/CFD: https://github.com/cuilimeng/CrackForest-dataset
- RDD2022: https://github.com/sekilab/RoadDamageDetector
- Mendeley 5y9wdsg2zt source appearance: https://data.mendeley.com/datasets/5y9wdsg2zt/2

## Dataset Catalog

| dataset | domain | size | annotation type | severity labels? | license | URL | status |
|---|---|---:|---|---|---|---|---|
| StreetSurfaceVis | pavement | 9,122 images in repo loader expectation | multi-label / classification for surface type and quality metadata | Repo-derived: quality labels used by Tarmac; not a verified standard severity scale | Not verified in research file | https://zenodo.org/api/records/11449977 | integrated |
| RTK | pavement | Not verified in research file | Not verified in research file; repo has Mendeley downloader | Not verified in research file | Not verified in research file | https://data.mendeley.com/datasets/fxy5khmhpb/1 | integrated |
| RSCD | pavement | Not verified in research file | Repo notes: surface material classification plus unevenness/friction annotations | Repo-derived: unevenness can approximate quality; not verified as severity | Not verified in research file | https://thu-rsxd.com/rscd/ and https://github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset | integrated |
| cracks_concrete_pavement / Mendeley 429vzbgmbx | concrete-generic | Not verified in research file | Repo-derived: binary classification, crack/non-crack | No verified severity labels | Not verified in research file | https://data.mendeley.com/datasets/429vzbgmbx/1 | integrated |
| runway_roboflow | runway | Not verified in research file | Repo-derived: COCO bounding boxes converted to crack tile labels; classes include crack/mildcrack/severecrack | Repo-derived: mild/severe class names may exist, but no verified standard mapping | Roboflow project terms; not verified in research file | https://universe.roboflow.com/revathi-deusp/runway-crack-detection-1iq1l | integrated |
| CrackAirport / Mendeley 3v5r2fxf89 | runway | Repo loader notes public page describes 2,226 examples; archive observed as 2,251 image/mask pairs | segmentation mask | No verified severity labels | Repo-derived: CC BY 4.0 | https://data.mendeley.com/datasets/3v5r2fxf89/1 | integrated |
| SDNET2018 | bridge / building / pavement / concrete-generic | over 56,000 256x256 images | classification, binary crack vs non-crack | No; verified as binary only, no bbox or segmentation masks | Not verified in research file | https://ieee-dataport.org/documents/sdnet2018-concrete-crack-image-dataset-machine-learning-applications | candidate |
| CODEBRIM | bridge / concrete-generic | Not verified in research file | multi-label / multi-target multi-class concrete defect classification | Defect labels yes; severity labels not verified | Not verified in research file | https://zenodo.org/records/2620293 | candidate |
| CrackForest / CFD | pavement | Not verified in research file | segmentation mask / pixel-level ground truth | No verified severity labels | Not verified in research file | https://github.com/cuilimeng/CrackForest-dataset | candidate |
| RDD2022 | pavement | Not verified in research file | bbox, Pascal VOC | No verified severity labels; four classes are D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole | CC BY-SA 4.0 | https://github.com/sekilab/RoadDamageDetector | candidate |
| Mendeley 5y9wdsg2zt | concrete-generic | Not verified in confirmed claims | Not verified in confirmed claims | Not verified in confirmed claims | Not verified in confirmed claims | https://data.mendeley.com/datasets/5y9wdsg2zt/2 | candidate |

## Verified Candidate Notes

SDNET2018 gives broad concrete crack/non-crack coverage across bridge decks, walls, and pavements, but it is binary classification only. It is useful for domain diversity and negative examples, not for crack geometry or severity.

CODEBRIM is bridge-sourced concrete defect data for multi-target multi-class classification, so it is the strongest verified candidate for moving beyond crack-only labels toward bridge/building concrete defects such as spalling-like or exposed-material categories where labels are present in the dataset.

CrackForest/CFD provides road crack pixel masks, which makes it a useful pavement segmentation candidate. RDD2022 provides object-detection boxes in Pascal VOC format for four road damage classes and is licensed CC BY-SA 4.0.

## Prioritized Acquisition Order

1. CODEBRIM: add next for bridge concrete multi-defect classification. It fills the largest current gap: multi-target structural defect labels beyond crack/non-crack.
2. SDNET2018: add after CODEBRIM for broad concrete crack/domain coverage across bridge decks, walls, and pavements. Its limitation is binary labels only.
3. CrackForest/CFD: add for road crack segmentation masks to complement CrackAirport runway masks with urban pavement masks.
4. RDD2022: add for object detection on pavement damage classes, especially longitudinal/transverse/alligator crack boxes and potholes under CC BY-SA 4.0.
5. Mendeley 5y9wdsg2zt: inspect manually before integration because it appears in the verified source list, but no confirmed claim establishes its size, annotations, license, or label schema.

**Engineering interpretation:** For bridges/buildings, CODEBRIM should come before SDNET2018 because multi-defect labels teach more structural-condition vocabulary. SDNET2018 should still be valuable for contrastive pretraining and crack robustness, but it cannot supervise masks, boxes, or severity. For pavement segmentation, CrackForest/CFD should precede RDD2022 if the immediate goal is pixel-level crack geometry; RDD2022 should precede CFD if the immediate goal is YOLO-style road-damage detection.
