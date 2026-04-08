import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# Diretório base do projeto (pasta pai de src/)
BASE_DIR = Path(__file__).parent.parent

def train_model():
    print("Carregando os dados...")
    df = pd.read_csv(BASE_DIR / 'data' / 'industrial_maintenance_data.csv')
    
    # Separando features e target
    X = df.drop(columns=['failure', 'machine_id'])
    y = df['failure']

    # MELHORIA 1: Adicionado 'stratify=y'
    # Isso garante que a proporção de 80/20 de falhas seja mantida tanto no treino quanto no teste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Treinando o modelo...")
    # MELHORIA 2: Adicionado 'class_weight="balanced"'
    # Isso penaliza o modelo mais severamente se ele errar a classe minoritária (as falhas),
    # melhorando potencialmente o Recall.
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    print("\nRelatório de Performance:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # MELHORIA 3: Extraindo e exibindo a Importância das Variáveis (Feature Importance)
    print("\nImportância dos Sensores (Feature Importance):")
    importances = model.feature_importances_
    for feature, imp in zip(X.columns, importances):
        print(f" - {feature}: {imp:.4f}")

    MODEL_PATH = Path(__file__).parent / 'maintenance_model.pkl'
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo salvo com sucesso em {MODEL_PATH}")

if __name__ == "__main__":
    train_model()