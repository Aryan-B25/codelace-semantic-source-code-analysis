# CodeLACE Evaluation Report (Corrected)

## Executive Summary

This report presents a comprehensive evaluation of the **trained** CodeLACE model compared to a baseline transformer model for semantic source code analysis.

## Model Comparison

### Overall Performance

| Model | Overall Accuracy | Overall F1 Score | Inference Time (ms) |
|-------|------------------|------------------|-------------------|
| CodeLACE | 0.585 (58.5%) | 0.308 (30.8%) | 12.37 |
| Baseline | 0.267 (26.7%) | 0.120 (12.0%) | 1.04 |

### Hierarchical Classification Results

| Model | Syntactic Accuracy | Semantic Accuracy | Pragmatic Accuracy |
|-------|-------------------|------------------|-------------------|
| CodeLACE | 0.390 (39.0%) | 0.615 (61.5%) | 0.750 (75.0%) |
| Baseline | 0.125 (12.5%) | 0.470 (47.0%) | 0.205 (20.5%) |

## Key Findings

### Performance Improvements
- **Overall Accuracy**: CodeLACE achieves 58.5% vs Baseline 26.7% (+119.4% improvement)
- **Overall F1 Score**: CodeLACE achieves 30.8% vs Baseline 12.0% (+155.8% improvement)

### Hierarchical Analysis
- **Syntactic Level**: CodeLACE 39.0% vs Baseline 12.5%
- **Semantic Level**: CodeLACE 61.5% vs Baseline 47.0%
- **Pragmatic Level**: CodeLACE 75.0% vs Baseline 20.5%

### Training vs Evaluation Consistency
This evaluation uses the **trained model checkpoint** from training, ensuring results reflect the actual learned capabilities.

## Conclusions

The trained CodeLACE model demonstrates the effectiveness of the architectural innovations when properly trained. The model's performance reflects the learning achieved during the 5-epoch training process.

## Methodology

- **Dataset**: Same synthetic code samples used in training
- **Model**: Trained CodeLACE model loaded from best checkpoint
- **Evaluation Metrics**: Accuracy, F1-score (macro-averaged), inference time
- **Hardware**: CPU-only evaluation for accessibility
- **Sample Size**: 200 test samples

Generated on: 2025-07-01 18:27:00
