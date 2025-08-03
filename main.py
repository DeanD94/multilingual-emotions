
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, f1_score, precision_score, recall_score
from scipy.stats import pearsonr
import os
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)

# parameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
NUM_LABELS = 6
LABEL_COLUMNS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


class EmotionDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe[LABEL_COLUMNS].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# model definition
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        x = self.classifier(x)
        return x

# handle multiple language .csv files
def load_multilingual_data(folder_path, filter_lang=None):
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            lang_code = file.split(".")[0]
            if filter_lang is None or lang_code == filter_lang:
                df = pd.read_csv(os.path.join(folder_path, file))
                df["lang"] = lang_code

                # Ensure all emotion columns exist
                for col in LABEL_COLUMNS:
                    if col not in df.columns:
                        df[col] = 0.0

                # normalize without clipping so we don't distort our data
                df[LABEL_COLUMNS] = df[LABEL_COLUMNS] / 3.0

                dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()



# evaluation Metric
def pearson_score(preds, labels):
    scores = []
    for i in range(NUM_LABELS):
        r, _ = pearsonr(preds[:, i], labels[:, i])
        scores.append(r)
    return np.mean(scores)

# train and evaluate model
def train_model(train_df, test_df):
    train_dataset = EmotionDataset(train_df)
    test_dataset = EmotionDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = EmotionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Check for invalid label values
            if torch.any(labels < 0) or torch.any(labels > 1):
                print("Out-of-bound label detected")
                print("Min label value:", labels.min().item(), "Max label value:", labels.max().item())
                print("Sample labels:", labels)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # Check if output is raw logits (should be, don't apply sigmoid here)
            if torch.isnan(outputs).any():
                print("NaN detected in outputs")
                print("Outputs:", outputs)

            loss = loss_fn(outputs, labels)

            # Check for negative loss
            if loss.item() < 0:
                print("Negative loss detected:", loss.item())
                print("Sample outputs:", outputs[:2].detach().cpu().numpy())
                print("Sample labels:", labels[:2].detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # evaluate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy() 
            all_preds.append(probs)
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # thresholded for F1 -> lower threshold for surprise/disgust
    thresholds = np.array([0.3, 0.4, 0.5, 0.5, 0.5, 0.3])
    thresholded_preds = (all_preds >= thresholds).astype(int)
    thresholded_labels = (all_labels >= thresholds).astype(int)
    
    # calculate evaluation metrics for each emotion. this will let us visualize
    per_emotion_f1 = f1_score(thresholded_labels, thresholded_preds, average=None)
    per_emotion_precision = precision_score(thresholded_labels, thresholded_preds, average=None)
    per_emotion_recall = recall_score(thresholded_labels, thresholded_preds, average=None)

    print("Metrics for each emotion:")
    for i, label in enumerate(LABEL_COLUMNS):
        print(f"{label:<10} | F1: {per_emotion_f1[i]:.3f} | Precision: {per_emotion_precision[i]:.3f} | Recall: {per_emotion_recall[i]:.3f}")

    # create a plot of f1 score for each emotion
    plt.figure(figsize=(10, 5))
    plt.bar(LABEL_COLUMNS, per_emotion_f1, color='skyblue')
    plt.title("Per-Emotion F1 Score")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    mse = mean_squared_error(all_labels, all_preds)
    pearson = pearson_score(all_preds, all_labels)
    macro_f1 = f1_score(thresholded_labels, thresholded_preds, average='macro')

    print(f"Test MSE: {mse:.4f}")
    print(f"Test Pearson r: {pearson:.4f}")
    print(f"Macro F1 (threshold=0.5): {macro_f1:.4f}")




if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "task-dataset", "semeval-2025-task11-dataset", "track_b", "train")
    test_path = os.path.join(base_dir, "task-dataset", "semeval-2025-task11-dataset", "track_b", "test")

    # modify language to test different multi-language combinations
    train_lang = "eng"
    test_lang = "rus"

    print(f"Training on: {train_lang}")
    print(f"Testing on: {test_lang}")

    train_df = load_multilingual_data(train_path, filter_lang=train_lang)
    test_df = load_multilingual_data(test_path, filter_lang=test_lang)

    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(test_df)} test samples")

    print(f"Sum of column labels in test(rus): {test_df[LABEL_COLUMNS].sum()}")
    train_model(train_df, test_df)
