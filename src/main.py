import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model():
    # Carregar
    df = pd.read_csv('data/industrial_maintenance_data.csv')
    X = df.drop(columns=['failure', 'machine_id'])
    y = df['failure']

    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Avaliar
    print("Relatório de Performance:")
    print(classification_report(y_test, model.predict(X_test)))

    # Guardar o modelo para uso futuro
    joblib.dump(model, 'src/maintenance_model.pkl')
    print("Modelo guardado em src/maintenance_model.pkl")

if __name__ == "__main__":
    train_model()