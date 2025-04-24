import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import re

# Step 1: Load your test results CSV
csv_path = 'youtube_video_classification.csv'  # Update this path if needed
df = pd.read_csv(csv_path)

# Step 2: Convert categorical labels to binary format
label_map = {'Safe': 0, 'Restricted': 1}
df['Actual Label'] = df['Actual Label'].map(label_map)
df['Predicted Label'] = df['Predicted Label'].map(label_map)

# Step 3: Extract actual and predicted labels
y_true = df['Actual Label']
y_pred = df['Predicted Label']

# Step 4: Calculate performance metrics
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Safe', 'Restricted'])

print("✅ Confusion Matrix:\n", conf_matrix)
print("\n📊 Classification Report:\n", report)

# Step 5: Parse the classification report for metrics
accuracy = float(re.findall(r'accuracy\s+([\d.]+)', report)[0]) * 100
precision = float(re.findall(r'weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', report)[0]) * 100
recall = float(re.findall(r'weighted avg\s+[\d.]+\s+([\d.]+)', report)[0]) * 100
f1_score = float(re.findall(r'weighted avg\s+([\d.]+)', report)[0]) * 100

# Step 6: Bar chart visualization
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1_score]
colors = ['skyblue', 'lightgreen', 'salmon', 'orchid']

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=colors)

# Add text labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5,
             f'{score:.1f}%', ha='center', fontsize=12, color='black')

plt.title('Performance Metrics of YouTube Age Classification Model', fontsize=14)
plt.ylabel('Score (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
