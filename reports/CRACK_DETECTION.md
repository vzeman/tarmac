# Crack Detection

Crack detection is trained and evaluated as a separate binary track from quality grading. This run includes runway-specific Roboflow imagery (`runway_roboflow`) alongside the concrete/pavement crack dataset.

Chosen threshold: `0.215` (max F1 on validation).

| Split | Source | Precision | Recall | F1 | Accuracy | ROC-AUC | Count | Positives |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | overall | 0.9987 | 0.9991 | 0.9989 | 0.9989 | 0.9998 | 4550 | 2286 |
| val | runway_roboflow | 0.8750 | 0.9545 | 0.9130 | 0.8889 | 0.9156 | 36 | 22 |
| val | concrete_pavement | 1.0000 | 0.9996 | 0.9998 | 0.9998 | 0.9999 | 4500 | 2250 |
| test | overall | 0.9987 | 0.9982 | 0.9985 | 0.9985 | 0.9997 | 4549 | 2284 |
| test | runway_roboflow | 0.8696 | 0.9524 | 0.9091 | 0.8889 | 0.9841 | 36 | 21 |
| test | concrete_pavement | 1.0000 | 0.9987 | 0.9993 | 0.9993 | 0.9997 | 4500 | 2250 |

