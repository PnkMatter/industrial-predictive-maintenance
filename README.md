# ⚙️ Industrial Predictive Maintenance AI

> A Machine Learning system that monitors industrial sensor data in real time and predicts equipment failures **before they happen** — with an interactive web dashboard powered by Streamlit.

---

## 📌 Problem Statement

Unplanned machinery downtime can cost manufacturing plants **thousands of dollars per hour**. Traditional maintenance follows fixed schedules, which leads to two problems:

- **Over-maintenance**: replacing healthy parts unnecessarily, wasting resources.
- **Under-maintenance**: missing failures that happen between scheduled check-ups.

This project solves both by using a trained ML model to continuously analyze sensor readings and trigger alerts only when a failure is genuinely imminent.

---

## 🚀 Features

- **Synthetic Data Generation**: Simulates 1,000 rows of realistic industrial sensor data (temperature, vibration, pressure, working hours) using statistical distributions and intentional noise.
- **Random Forest Classifier**: Ensemble model that handles imbalanced classes and provides feature importance scores out of the box.
- **Model Improvements** (v2):
  - `stratify=y` in train/test split — preserves the failure ratio in both sets, preventing biased evaluation.
  - `class_weight='balanced'` — penalizes misclassifications of the minority class (failures) more heavily, improving Recall.
  - Feature importance report printed to the console after every training run.
- **Interactive Web Dashboard** (`src/app.py`): A Streamlit interface that lets you simulate sensor readings via sliders and run a live diagnostic with the trained model.
- **Path-agnostic execution**: Both `main.py` and `app.py` use `pathlib.Path(__file__)` so they resolve files correctly regardless of the directory the command is run from.

---

## 🧠 Model Architecture

| Component          | Choice                          | Reason                                                                 |
|--------------------|---------------------------------|------------------------------------------------------------------------|
| Algorithm          | Random Forest Classifier        | Robust, handles noise well, native feature importance                  |
| Train/Test Split   | 80% / 20% with `stratify=y`     | Guarantees proportional failure representation in the test set         |
| Class Weighting    | `class_weight='balanced'`       | Compensates for the naturally low frequency of failures in the dataset |
| Serialization      | `joblib`                        | Efficient and standard for scikit-learn models                         |

### Sensor Features Used

| Feature         | Description                      | Failure Threshold |
|-----------------|----------------------------------|-------------------|
| `temperature`   | Machine temperature in °C        | > 85°C            |
| `vibration`     | Vibration intensity in mm/s      | > 8 mm/s          |
| `pressure`      | Operating pressure in PSI        | —                 |
| `working_hours` | Cumulative machine runtime in h  | —                 |

---

## 📊 Results & Performance

Model evaluated on 200 samples (20% hold-out set, never seen during training):

```text
              precision    recall  f1-score   support

           0       0.96      0.99      0.98       166   ← Healthy
           1       0.96      0.79      0.87        34   ← Failure

    accuracy                           0.96       200
   macro avg       0.96      0.89      0.92       200
weighted avg       0.96      0.96      0.96       200
```

### Sensor Importance (Feature Importance)

```text
 - temperature:   0.3724  (37.2%)
 - vibration:     0.4308  (43.1%)  ← Most predictive
 - pressure:      0.1027  (10.3%)
 - working_hours: 0.0942   (9.4%)
```

### Interpretation

- **Precision (96%)**: When the model predicts a failure, it is correct 96% of the time — minimizing false alarms and wasted maintenance trips.
- **Recall (79%)**: The model catches ~80% of all actual failures. Some are missed due to intentional noise in the synthetic data simulating random hardware glitches — acceptable for a baseline.
- **F1-Score (0.87)**: Strong balance between precision and recall, confirming the model is reliable for production decision-making.

---

## 🖥️ Interactive Dashboard (Streamlit)

The web interface (`src/app.py`) allows operators to:

1. Adjust sensor readings in real time via **sidebar sliders**:
   - 🌡️ Temperature (°C)
   - 📳 Vibration (mm/s)
   - 💨 Pressure (PSI)
   - ⏱️ Working Hours (h)

2. Click **"Executar Diagnóstico"** to run an inference with the trained model.

3. Instantly see the predicted status:
   - ✅ **Operação Normal** — with the estimated failure probability.
   - ⚠️ **ALERTA: Risco de Falha Detectado!** — with a recommended action.

---

## 📁 Project Structure

```text
industrial-predictive-maintenance/
├── data/
│   └── industrial_maintenance_data.csv   # Synthetic sensor dataset (1,000 rows)
├── src/
│   ├── main.py                           # Model training pipeline
│   ├── app.py                            # Streamlit web dashboard
│   └── maintenance_model.pkl             # Serialized trained model (generated)
├── generate_data.py                      # Synthetic data generator
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## ⚙️ How to Run

### Prerequisites

- Python 3.9+
- A virtual environment (recommended)

### 1. Clone & set up the environment

```bash
git clone <your-repo-url>
cd industrial-predictive-maintenance

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic dataset

```bash
python generate_data.py
```

> This creates `data/industrial_maintenance_data.csv` with 1,000 rows of simulated sensor readings.

### 4. Train the model

```bash
python src/main.py
```

> This trains the Random Forest, prints the classification report and feature importances, and saves the model to `src/maintenance_model.pkl`.

### 5. Launch the dashboard

```bash
.venv\Scripts\streamlit.exe run src\app.py
```

> Open [http://localhost:8501](http://localhost:8501) in your browser. Use the sliders on the left sidebar to simulate sensor conditions and run a diagnostic.

---

## 📦 Dependencies

| Package        | Purpose                              |
|----------------|--------------------------------------|
| `pandas`       | Data loading and manipulation        |
| `numpy`        | Numerical operations in data generation |
| `scikit-learn` | Random Forest model and evaluation   |
| `joblib`       | Model serialization                  |
| `matplotlib`   | Plotting (available for notebooks)   |
| `seaborn`      | Statistical visualization            |
| `streamlit`    | Interactive web dashboard            |

---

## 🔭 Future Improvements

- [ ] Replace synthetic data with real sensor feeds via MQTT or REST API
- [ ] Add time-series visualization of sensor trends in the dashboard
- [ ] Implement anomaly detection as a complementary unsupervised layer
- [ ] Containerize with Docker for production deployment
- [ ] Add unit tests for the training pipeline

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.