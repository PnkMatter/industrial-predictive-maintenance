# 1. Industrial Predictive Maintenance AI

## Project Overview
This project demonstrates a Machine Learning solution designed to reduce unplanned downtime in manufacturing environments. By analyzing real-time sensor data, the system predicts equipment failures before they occur, allowing maintenance teams to transition from reactive to **predictive maintenance strategies**.

---

## Problem Statement
Unplanned machinery downtime can cost manufacturing plants thousands of dollars per hour. Traditional maintenance is often performed on fixed schedules, which can be inefficient—either replacing parts too early or failing to prevent a breakdown. This project aims to solve that by using data to pinpoint the exact moment an intervention is needed.

---

## Technical Implementation

* **Synthetic Data Generation**: Developed a Python-based simulator to generate industrial sensor data (Temperature, Vibration, Pressure) using normal distributions and statistical noise to mimic real-world unpredictability.
* **Machine Learning Model**: Implemented a **Random Forest Classifier**, an ensemble learning method that is highly effective for industrial data and provides insights into feature importance.
* **Data Pipeline**: Built a modular structure for data processing, model training, and performance evaluation, ensuring the code is scalable and maintainable.

---

## Results & Performance

The model was evaluated on a test set (20% of total data) that it had never seen before to ensure real-world applicability.

### Classification Report:
```text
Relatório de Performance:
              precision    recall  f1-score   support

           0       0.96      0.99      0.98       162
           1       0.97      0.82      0.89        38

    accuracy                           0.96       200
```

## Final Verdict & Conclusion

The model achieved an outstanding 96% overall Accuracy. However, in an industrial context, we look deeper into the specific metrics for Class 1 (Failures):

- Precision (97%): When the model predicts a failure, it is correct 97% of the time. This minimizes "False Alarms," ensuring maintenance teams don't waste time on healthy machines.

- Recall (82%): The model successfully identified 82% of all actual failures in the dataset. While some failures were missed (due to the intentional noise added to simulate random hardware glitches), catching over 80% of breakdowns significantly reduces operational risk.

- F1-Score (0.89): The high F1-score confirms a robust balance between precision and recall, making it a reliable tool for decision-making.

## Conclusion
This AI engine proves that sensor-based monitoring can effectively predict mechanical health. Implementing this solution in a real-world production line would result in lower maintenance costs, optimized spare part inventory, and significantly higher equipment availability (OEE).

## Project Structure
```text
industrial-predictive-maintenance/
├── data/
│   └── industrial_maintenance_data.csv
├── src/
│   ├── main.py
│   └── maintenance_model.pkl
├── requirements.txt
└── README.md
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**:
   ```bash
   python generate_data.py
   ```

3. **Train Model**:
   ```bash
   python src/main.py
   ```

4. **View Results**:
   Check the console output for the classification report and model performance metrics.