| measure | value |
|---|---|
| dev images (folds 0-3) | 566 |
| annotator records | 914 |
| GT instances | 6483 |
| predicted instances | 3705 |
| prediction x record pairs (FP base) | 6018 |
| PQ (notebook definition) | 0.4706 |
| PQ image-bootstrap SE | 0.0049 |
| SQ (mean IoU of TP pairs) | 0.6887 |
| RQ | 0.6833 |
| TP / FP / FN | 4271 / 1747 / 2212 |
| PQ per fold 0 / 1 / 2 / 3 | 0.4815 / 0.4542 / 0.4905 / 0.4578 |
| PQ, test-team records (02/03/05/06) | 0.4930 |
| PQ, teams 01/04 records | 0.4519 |
| PQ, analysis scorer (label-map GT, as hyg_read) | 0.470559 |
| overlapping pairs (IoU > 0) | 5315 |
| matched pairs (IoU > 0.5) | 4271 |
| IoU, all overlapping pairs: mean / median / IQR | 0.617 / 0.665 / 0.544-0.741 |
| IoU, matched pairs: mean / median / IQR | 0.689 / 0.697 / 0.624-0.756 |
| Dice, all overlapping pairs: mean / median / IQR | 0.746 / 0.798 / 0.705-0.851 |
| Dice, matched pairs: mean / median / IQR | 0.812 / 0.821 / 0.768-0.861 |
| many-to-many components (>= 2 GT and >= 2 predictions linked at IoU > 0) | 7 |
| GT instances overlapped by 0 (miss) prediction(s) [1:n] | 1323 |
| GT instances overlapped by 1 prediction(s) [1:n] | 5016 |
| GT instances overlapped by 2 prediction(s) [1:n] | 133 |
| GT instances overlapped by 3 prediction(s) [1:n] | 11 |
| GT instances overlapped by 4 or more prediction(s) [1:n] | 0 |
| predictions overlapping 0 (miss) GT instance(s) of a record [n:1] | 837 |
| predictions overlapping 1 GT instance(s) of a record [n:1] | 5051 |
| predictions overlapping 2 GT instance(s) of a record [n:1] | 126 |
| predictions overlapping 3 GT instance(s) of a record [n:1] | 4 |
| predictions overlapping 4 or more GT instance(s) of a record [n:1] | 0 |

Full degree counts. GT (1:n): 0: 1323, 1: 5016, 2: 133, 3: 11. Predictions (n:1): 0: 837, 1: 5051, 2: 126, 3: 4.
