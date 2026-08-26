# 📱 Mobile Price Range Classification

A machine learning project that predicts the **price range of a mobile phone** (Low, Medium, High, Very High) based on its technical specifications, using both a classical ML model (Random Forest) and a Deep Learning model (Neural Network built with Keras/TensorFlow).

---

## 📌 Overview

Choosing the right price for a new phone model is a critical decision for manufacturers and retailers. This project builds a classifier that takes in a phone's hardware specs (battery, RAM, camera, screen size, etc.) and predicts which price segment it belongs to.

Two different modeling approaches were built and compared:

| Model | Type | Test Accuracy |
|---|---|---|
| Random Forest Classifier | Classical ML | **89.25%** |
| Deep Neural Network (Keras) | Deep Learning | **90.00%** |

---

## 🗂️ Dataset

The dataset (`mobile.csv`) contains **2,000 rows** and **21 columns**, with no missing values. Each row represents a mobile phone with the following features:

| Feature | Description |
|---|---|
| `battery_power` | Total energy a battery can store (mAh) |
| `blue` | Has Bluetooth or not |
| `clock_speed` | Speed at which the microprocessor executes instructions |
| `dual_sim` | Has dual SIM support or not |
| `fc` | Front camera megapixels |
| `four_g` | Has 4G or not |
| `int_memory` | Internal memory (GB) |
| `m_dep` | Mobile depth (cm) |
| `mobile_wt` | Weight of mobile phone |
| `n_cores` | Number of processor cores |
| `pc` | Primary camera megapixels |
| `px_height` / `px_width` | Pixel resolution height/width |
| `ram` | RAM (MB) |
| `sc_h` / `sc_w` | Screen height/width (cm) |
| `talk_time` | Longest time a battery charge lasts (hours) |
| `three_g` | Has 3G or not |
| `touch_screen` | Has touchscreen or not |
| `wifi` | Has WiFi or not |
| `price_range` | **Target** — 0 (low), 1 (medium), 2 (high), 3 (very high) |

---

## ⚙️ Methodology

1. **Preprocessing**
   - Checked and confirmed no missing values.
   - Encoded any categorical columns with `LabelEncoder`.
   - Split data into train/test sets (80/20).
   - Scaled features with `StandardScaler` (fit on train, applied to test — no data leakage).

2. **Model 1 — Random Forest Classifier**
   - Trained with default hyperparameters (`random_state=42`).
   - Evaluated using accuracy and a full classification report (precision/recall/F1 per class).

3. **Model 2 — Neural Network (Keras Sequential)**
   - Architecture: `Dense(200) → Dropout(0.3) → Dense(150) → Dropout(0.3) → Dense(100) → Dropout(0.3) → Dense(4, softmax)`
   - Dropout layers added specifically to reduce overfitting (validated during development — train/validation/test accuracy stayed within ~1-2% of each other).
   - Loss: `sparse_categorical_crossentropy` (multi-class target, integer-encoded).
   - Optimizer: `Adam`.
   - Trained for 12 epochs, batch size 32, with a 20% validation split.

---

## 📊 Results

**Random Forest — Classification Report**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| 0 (Low) | 0.95 | 0.96 | 0.96 |
| 1 (Medium) | 0.89 | 0.87 | 0.88 |
| 2 (High) | 0.78 | 0.87 | 0.82 |
| 3 (Very High) | 0.94 | 0.87 | 0.90 |
| **Overall Accuracy** | | | **0.89** |

**Neural Network**

- Final Test Accuracy: **90.0%**
- Final Test Loss: **0.23**
- Training and validation accuracy stayed close throughout training (e.g., epoch 12: train 90.8% vs. validation 92.2%), indicating the model generalizes well and is **not overfitting**.

Both models perform comparably, with the neural network having a slight edge. The neural network also produces smoother class probabilities (via `softmax`), which can be useful in the business use case below.

---

## 💼 Business Use Case

A model like this has direct value in the mobile phone industry:

- **Product pricing strategy**: Before launch, a manufacturer can feed a prototype's specs into the model to see which price segment it naturally falls into — helping validate or adjust pricing decisions before going to market.
- **Competitive benchmarking**: Retailers and e-commerce platforms can use the model to auto-tag new phone listings into price tiers, even when the listed price is missing, inconsistent, or needs sanity-checking against the phone's actual specs.
- **Spec-to-price optimization**: R&D teams can use the model to answer "what-if" questions — e.g., how much does increasing RAM or battery capacity push a phone into a higher price bracket — to optimize the specs-to-price ratio for a target market segment.
- **Customer-facing tools**: A simplified version of this model could power a "what's a fair price for this phone?" feature in a shopping app, increasing customer trust.

In short, this isn't just an academic classification exercise — it's a decision-support tool for pricing and product strategy.

---

## 🚀 Tech Stack

- Python
- Pandas
- Scikit-learn (`RandomForestClassifier`, `StandardScaler`, `LabelEncoder`, `train_test_split`)
- TensorFlow / Keras (`Sequential`, `Dense`, `Dropout`)

---

## ▶️ How to Run

```bash
pip install pandas scikit-learn tensorflow
jupyter notebook code.ipynb
```

Make sure `mobile.csv` is in the same directory as the notebook.

---

## ✅ Conclusion

This project demonstrates a full, practical ML workflow — from raw tabular data to a deployable classifier — using two complementary approaches:

- The **Random Forest** model is a strong, fast, easily interpretable baseline (89.25% accuracy) that requires no scaling and gives clear feature-level insight via `classification_report`.
- The **Neural Network** slightly outperforms it (90% accuracy) and, with dropout regularization, generalizes reliably to unseen data — making it a solid choice if the dataset grows or more complex, non-linear patterns emerge in future data.

For a production setting, the Random Forest model would be a reasonable first choice given its simplicity and interpretability, while the neural network leaves more headroom for improvement as more data becomes available (e.g., via hyperparameter tuning, more epochs with early stopping, or a deeper architecture).

**Possible next steps:**
- Hyperparameter tuning (GridSearch/RandomSearch for Random Forest; learning rate/architecture search for the NN).
- Add `EarlyStopping` and `ModelCheckpoint` callbacks to the NN training loop.
- Feature importance analysis to identify which specs most influence price range.
- Deploy the best model behind a simple API (e.g., Flask/FastAPI) for real-time predictions.
