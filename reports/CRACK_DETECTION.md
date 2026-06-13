# Crack Detection

Crack detection is trained and evaluated as a separate binary track from quality grading.

Chosen threshold: `0.255` (max F1 on validation).

| Split | Precision | Recall | F1 | Accuracy | ROC-AUC | Count | Positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 1.0000 | 0.9991 | 0.9996 | 0.9996 | 1.0000 | 4513 | 2263 |
| test | 0.9991 | 0.9987 | 0.9989 | 0.9989 | 0.9996 | 4513 | 2263 |

