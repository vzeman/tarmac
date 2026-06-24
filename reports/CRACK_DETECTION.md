# Crack Detection

Crack detection is trained and evaluated as a separate binary track from quality grading. This run includes runway-specific Roboflow imagery (`runway_roboflow`) alongside the concrete/pavement crack dataset.

Chosen threshold: `0.485` (max F1 on validation).

| Split | Source | Precision | Recall | F1 | Accuracy | ROC-AUC | Count | Positives |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | overall | 0.9905 | 0.9686 | 0.9794 | 0.9794 | 0.9959 | 25148 | 12743 |
| val | runway_roboflow | 0.9000 | 0.4091 | 0.5625 | 0.6111 | 0.7208 | 36 | 22 |
| val | concrete_pavement | 1.0000 | 0.9969 | 0.9984 | 0.9984 | 0.9999 | 4500 | 2250 |
| test | overall | 0.9918 | 0.9714 | 0.9815 | 0.9814 | 0.9968 | 25146 | 12740 |
| test | runway_roboflow | 1.0000 | 0.6667 | 0.8000 | 0.8056 | 0.8381 | 36 | 21 |
| test | concrete_pavement | 1.0000 | 0.9951 | 0.9975 | 0.9976 | 0.9999 | 4500 | 2250 |

