# Classifier evaluation

## Summary

| Classifier | Macro F1 | Micro F1 | Micro P | Micro R | Exact match | Hamming loss | Jaccard |
|---|---|---|---|---|---|---|---|
| heuristic | 0.860 | 0.892 | 1.000 | 0.805 | 0.877 | 0.018 | 0.877 |
| llm | 0.789 | 0.795 | 0.838 | 0.756 | 0.737 | 0.035 | 0.781 |
| ensemble | 0.885 | 0.900 | 0.923 | 0.878 | 0.860 | 0.018 | 0.877 |
| ensemble+review | 0.937 | 0.949 | 1.000 | 0.902 | 0.930 | 0.009 | 0.930 |

## Per-category F1

| Category | Support | heuristic P/R/F1 | llm P/R/F1 | ensemble P/R/F1 | ensemble+review P/R/F1 |
|---|---|---|---|---|---|
| hallucination | 5 | 1.00 / 0.60 / 0.75 | 1.00 / 0.60 / 0.75 | 1.00 / 0.80 / 0.89 | 1.00 / 0.80 / 0.89 |
| factual_error | 5 | 1.00 / 0.40 / 0.57 | 1.00 / 0.40 / 0.57 | 1.00 / 0.40 / 0.57 | 1.00 / 0.60 / 0.75 |
| omission | 4 | 1.00 / 1.00 / 1.00 | 1.00 / 0.75 / 0.86 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| reasoning_error | 4 | 1.00 / 0.50 / 0.67 | 1.00 / 0.75 / 0.86 | 1.00 / 0.75 / 0.86 | 1.00 / 0.75 / 0.86 |
| formatting_failure | 5 | 1.00 / 1.00 / 1.00 | 0.71 / 1.00 / 0.83 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| over_refusal | 6 | 1.00 / 1.00 / 1.00 | 0.71 / 0.83 / 0.77 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 |
| instruction_violation | 7 | 1.00 / 1.00 / 1.00 | 0.83 / 0.71 / 0.77 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| unsupported_claim | 5 | 1.00 / 0.80 / 0.89 | 0.83 / 1.00 / 0.91 | 0.83 / 1.00 / 0.91 | 1.00 / 1.00 / 1.00 |

## Multi-label confusion (ensemble+review)

Rows are gold labels, columns are predicted labels; a cell counts items carrying both.

| gold \ pred | halluc | factua | omissi | reason | format | over | instru | unsupp | none |
|---|---|---|---|---|---|---|---|---|---|
| hallucination | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 |
| factual_error | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 2 |
| omission | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| reasoning_error | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 1 |
| formatting_failure | 0 | 0 | 0 | 0 | 5 | 0 | 1 | 0 | 0 |
| over_refusal | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| instruction_violation | 0 | 1 | 0 | 0 | 1 | 0 | 7 | 0 | 0 |
| unsupported_claim | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
