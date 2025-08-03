# SemEval 2025 Task 11: Multilingual Emotion Intensity Prediction
## Problem Definition
This project addresses SemEval 2025 Task 11, which involves predicting the intensity of six emotions (anger, disgust, fear, joy, sadness, surprise) from multilingual news headlines. We explore both regression-based multi-label prediction and cross-cultural transfer learning across languages.

## Dataset Used
We use the official dataset from SemEval 2025 Task 11, Track B and Track C. It consists of annotated news headlines in multiple languages, with each headline scored for the intensity of six emotions.

Label columns: anger, disgust, fear, joy, sadness, surprise

Languages include: English (eng), Russian (rus), among others

Emotion intensities range from 0 to 3 and are normalized to 0–1 before training

Track B provides multilingual headlines with corresponding emotion scores. Track C enables transfer learning scenarios, allowing models to train on one language and evaluate on another (cross-lingual emotion prediction).

## Evaluation Metrics
We evaluate model performance using the following metrics:

Pearson Correlation Coefficient (r): Measures the linear correlation between predicted and true emotion scores (main evaluation metric)

Mean Squared Error (MSE): Evaluates prediction accuracy for continuous emotion intensities

Macro F1 Score: Used after thresholding predicted values to binary labels (0/1)

Per-emotion Precision, Recall, and F1 Scores: For detailed analysis of performance on each emotion class

We also visualize the per-emotion F1 scores to highlight emotion-specific gaps.

## Model Explanation
We fine-tune a pretrained multilingual transformer model (xlm-roberta-base) for multi-label regression.

## Architecture:
Tokenization via AutoTokenizer

Transformer backbone via AutoModel

The [CLS] token embedding is passed through a dropout layer and a linear regression head

Output: Six continuous values (one per emotion)

## Training Setup:
Loss function: BCEWithLogitsLoss

Optimizer: AdamW

Batch size: 16

Epochs: 3

Learning rate: 2e-5

The model is trained on emotion intensities normalized between 0–1.

## Results and Baseline Comparison
### Experimental Setup:
Train language: English (eng)

Test language: Russian (rus)

Training samples: 2768

Test samples: 650

Per-Emotion Scores (F1 | Precision | Recall):
anger: 0.512 | 0.598 | 0.448

disgust: 0.000 | 0.000 | 0.000

fear: 0.417 | 0.267 | 0.947

joy: 0.391 | 0.587 | 0.293

sadness: 0.516 | 0.449 | 0.606

surprise: 0.407 | 0.302 | 0.625

Aggregated Scores:
Macro F1: 0.374

Mean Squared Error (MSE): 0.155

Pearson Correlation (r): 0.401

According to the SemEval 2025 baseline notebook:

Baseline Pearson r for English: ~0.78

Baseline Pearson r for Russian: ~0.68

Our result (0.4012) reflects zero-shot performance, since no fine-tuning is done on the Russian data. This demonstrates moderate success in emotion transfer across languages using a shared transformer backbone without adapters.

## Future Improvements
Add language-specific adapters (e.g., via AdapterHub) to capture culture-specific emotion expression. We attempted to use adapter modules but faced compatibility issues with AutoModelWithHeads.

Improve the model's ability to predict difficult emotions like disgust, which showed zero F1 due to low prevalence and class imbalance.

Explore loss functions better suited for ordinal regression tasks, such as MSE or margin-based losses.

Perform few-shot tuning with a small number of target-language samples.

Incorporate multilingual data augmentation such as translation and paraphrasing to enhance model generalization.

## Key Takeaways
Zero-shot transfer learning from English to Russian is viable and achieves reasonable emotion prediction performance, especially for sadness, anger, and surprise.

The model's inability to detect emotions like disgust may be due to sparse labels and insufficient cross-lingual generalization.

Cross-cultural emotion perception remains a challenge and benefits from language-specific tuning and augmentation.

Despite not using adapters or few-shot learning, the model demonstrates potential for multilingual affective computing tasks.

