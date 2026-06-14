# Data Licenses And Commercial-Use Labels

This is an engineering license inventory for Tarmac capabilities, not legal advice. Confirm obligations with counsel before redistributing data, weights, or commercial products built from the datasets.

## Capability Labels

| Capability | Datasets currently used | Commercial-use label | Notes |
|---|---|---|---|
| Surface type + 1-5 quality grading | StreetSurfaceVis, Zenodo `11449977` | Commercial-usable CC data | Zenodo lists StreetSurfaceVis as **Creative Commons Attribution Share Alike 4.0 International**. Commercial use is allowed under CC BY-SA 4.0, with attribution and ShareAlike obligations. |
| Crack detection/classification | CrackAirport, Mendeley `3v5r2fxf89`; cracks_concrete_pavement, Mendeley `429vzbgmbx` | Commercial-usable CC data | Mendeley lists CrackAirport under **CC BY 4.0**. The Mendeley cracks_concrete_pavement license field is **CC BY 4.0**; preserve attribution and verify any dataset-page usage text before redistribution. |
| Crack segmentation/geometry examples and masks | CrackAirport, Mendeley `3v5r2fxf89` | Commercial-usable CC data | CrackAirport is the committed license-safe crack imagery source for examples and smoke imagery. |
| Multi-label structural defect head: `crack` | Mix includes CC crack datasets plus CODEBRIM-derived structural examples | Mixed; use caution | The crack label is also supported by CC crack datasets, but the unified defect head is trained jointly with CODEBRIM examples. Treat the full defect-head checkpoint as mixed-license unless retrained on commercial-safe sources only. |
| Multi-label structural defect head: `spalling`, `efflorescence`, `exposed_rebar`, `corrosion` | CODEBRIM, Zenodo `2620293` | **NON-COMMERCIAL / research-only** | CODEBRIM's Zenodo record reports license id `other-nc`. These non-crack labels should not be represented as commercial-safe. |
| SDNET2018 crack/non-crack candidate data | SDNET2018, IEEE DataPort | Unverified | License status is not verified in the local research notes. Do not label SDNET2018-derived training or outputs as commercial-safe until the governing license is confirmed. |

## Source Links

- StreetSurfaceVis: https://zenodo.org/records/11449977
- CrackAirport: https://data.mendeley.com/datasets/3v5r2fxf89/1
- cracks_concrete_pavement: https://data.mendeley.com/datasets/429vzbgmbx/1
- CODEBRIM: https://zenodo.org/records/2620293
- SDNET2018: https://ieee-dataport.org/documents/sdnet2018-concrete-crack-image-dataset-machine-learning-applications

## Practical Rule

Use the surface-quality and crack-specific capabilities as the commercial-safe path, subject to CC attribution/share-alike requirements. Treat the full structural defect head, and especially its non-crack outputs (`spalling`, `efflorescence`, `exposed_rebar`, `corrosion`), as non-commercial/research-only unless retrained on verified commercial-safe data.
