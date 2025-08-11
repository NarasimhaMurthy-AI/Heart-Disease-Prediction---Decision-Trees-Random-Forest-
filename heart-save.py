import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# === Load Data ===
df = pd.read_csv("heart.csv")  # Change path if needed

# === Features & Target ===
X = df.drop('target', axis=1)
y = df['target']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Decision Tree (Full Depth) ===
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(18, 9))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'],
          filled=True, fontsize=7)
plt.title("Decision Tree (Full Depth)")
plt.show()
# plt.savefig("decision_tree_full.png", dpi=300, bbox_inches="tight")

# === Accuracy vs. max_depth sweep ===
depths = list(range(1, 13))
train_acc, test_acc, cv_mean = [], [], []
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))
    cv_mean.append(cross_val_score(model, X, y, cv=5).mean())

plt.figure(figsize=(8, 5))
plt.plot(depths, train_acc, label='Train Accuracy', marker='o')
plt.plot(depths, test_acc, label='Test Accuracy', marker='o')
plt.plot(depths, cv_mean, label='CV Mean Accuracy', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("accuracy_vs_depth.png", dpi=300, bbox_inches="tight")

# === Decision Tree (max_depth=10) ===
dt_lim = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_lim.fit(X_train, y_train)

plt.figure(figsize=(18, 9))
plot_tree(dt_lim, feature_names=X.columns, class_names=['No Disease', 'Disease'],
          filled=True, fontsize=7)
plt.title("Decision Tree (Max Depth = 10)")
plt.show()
# plt.savefig("decision_tree_depth10.png", dpi=300, bbox_inches="tight")

# === Random Forest ===
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Feature Importances Bar Chart
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

top10 = importances.head(10)
plt.figure(figsize=(8, 5))
plt.barh(top10['feature'], top10['importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.show()
# plt.savefig("top10_features.png", dpi=300, bbox_inches="tight")

# === ROC Curves ===
models = {
    "Decision Tree (Full)": dt,
    "Decision Tree (Depth=10)": dt_lim,
    "Random Forest": rf
}

plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("roc_curves.png", dpi=300, bbox_inches="tight")

# === Print Summary ===
print("\nModel Accuracies:")
print(f"Decision Tree (Full): Train={accuracy_score(y_train, dt.predict(X_train)):.4f}, "
      f"Test={accuracy_score(y_test, dt.predict(X_test)):.4f}")
print(f"Decision Tree (Depth=10): Train={accuracy_score(y_train, dt_lim.predict(X_train)):.4f}, "
      f"Test={accuracy_score(y_test, dt_lim.predict(X_test)):.4f}")
print(f"Random Forest: Train={accuracy_score(y_train, rf.predict(X_train)):.4f}, "
      f"Test={accuracy_score(y_test, rf.predict(X_test)):.4f}")
input("Press Enter to close...")