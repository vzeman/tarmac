# Structural Defect Detection

> Non-commercial banner: the non-crack defect labels (`spalling`, `efflorescence`, `exposed_rebar`, `corrosion`) are trained on CODEBRIM (Zenodo `2620293`, license id `other-nc`) and should be treated as research-only/non-commercial.

The defect head is a multi-label classifier trained on frozen active-backbone DINOv3 embeddings. It predicts crack, spalling, efflorescence, exposed rebar, and corrosion; pure `none` examples are used only as negatives.

Checkpoint: `models/defect_head.pt`
Embeddings: `data/processed/defect_embeddings.parquet`
Best validation macro-AP: `0.9654`

Thresholds are chosen per label by maximizing F1 on the validation split.

| Label | Threshold |
| --- | ---: |
| crack | 0.420 |
| spalling | 0.950 |
| efflorescence | 0.950 |
| exposed_rebar | 0.875 |
| corrosion | 0.895 |

## Val Per-label Metrics

| Label | Precision | Recall | F1 | AP | Support | Predicted + |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crack | 0.9569 | 0.9231 | 0.9397 | 0.9863 | 4160 | 4013 |
| spalling | 0.9120 | 0.8881 | 0.8999 | 0.9644 | 420 | 409 |
| efflorescence | 0.9482 | 0.9581 | 0.9531 | 0.9795 | 382 | 386 |
| exposed_rebar | 0.9470 | 0.9715 | 0.9591 | 0.9878 | 386 | 396 |
| corrosion | 0.8248 | 0.9370 | 0.8774 | 0.9090 | 397 | 451 |

## Val Per-domain Metrics

| Domain | Rows | Positive Labels | Macro Labels | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Macro AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 7891 | 5745 | 5 | 0.9178 | 0.9356 | 0.9258 | 0.9344 | 0.9654 |
| bridge | 2087 | 2318 | 5 | 0.9037 | 0.9149 | 0.9083 | 0.9000 | 0.9551 |
| building | 1261 | 596 | 1 | 0.8717 | 0.8646 | 0.8662 | 0.8526 | 0.9366 |
| concrete_generic | 2930 | 2250 | 1 | 0.9964 | 0.9969 | 0.9967 | 0.9984 | 0.9999 |
| pavement | 1383 | 398 | 1 | 0.9199 | 0.8973 | 0.9076 | 0.8661 | 0.9380 |
| runway | 230 | 183 | 1 | 0.9532 | 0.9386 | 0.9457 | 0.9783 | 0.9959 |

## Val Multi-label Summary

| Metric | Value |
| --- | ---: |
| rows | 7891 |
| exact_match | 0.9140 |
| true_label_assignments | 5745 |
| predicted_label_assignments | 5655 |
| true_positive_assignments | 5326 |
| false_positive_assignments | 329 |
| false_negative_assignments | 419 |
| images_with_any_true_defect | 4890 |
| images_with_any_predicted_defect | 4734 |
| mean_true_labels_per_image | 0.7280 |
| mean_predicted_labels_per_image | 0.7166 |

## Test Per-label Metrics

| Label | Precision | Recall | F1 | AP | Support | Predicted + |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crack | 0.9582 | 0.9212 | 0.9393 | 0.9868 | 4160 | 3999 |
| spalling | 0.8991 | 0.8915 | 0.8953 | 0.9660 | 470 | 466 |
| efflorescence | 0.9147 | 0.9511 | 0.9325 | 0.9675 | 327 | 340 |
| exposed_rebar | 0.9360 | 0.9838 | 0.9593 | 0.9863 | 431 | 453 |
| corrosion | 0.8047 | 0.9036 | 0.8513 | 0.8982 | 415 | 466 |

## Test Per-domain Metrics

| Domain | Rows | Positive Labels | Macro Labels | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Macro AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 7889 | 5803 | 5 | 0.9026 | 0.9302 | 0.9155 | 0.9302 | 0.9610 |
| bridge | 2132 | 2402 | 5 | 0.8897 | 0.9078 | 0.8975 | 0.8903 | 0.9507 |
| building | 1234 | 580 | 1 | 0.8845 | 0.8747 | 0.8768 | 0.8626 | 0.9461 |
| concrete_generic | 2930 | 2250 | 1 | 0.9993 | 0.9978 | 0.9986 | 0.9993 | 1.0000 |
| pavement | 1362 | 387 | 1 | 0.9085 | 0.8919 | 0.8996 | 0.8545 | 0.9270 |
| runway | 231 | 184 | 1 | 0.9023 | 0.8959 | 0.8990 | 0.9593 | 0.9959 |

## Test Multi-label Summary

| Metric | Value |
| --- | ---: |
| rows | 7889 |
| exact_match | 0.9089 |
| true_label_assignments | 5803 |
| predicted_label_assignments | 5724 |
| true_positive_assignments | 5361 |
| false_positive_assignments | 363 |
| false_negative_assignments | 442 |
| images_with_any_true_defect | 4887 |
| images_with_any_predicted_defect | 4727 |
| mean_true_labels_per_image | 0.7356 |
| mean_predicted_labels_per_image | 0.7256 |
