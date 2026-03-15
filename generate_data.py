import pandas as pd
import numpy as np

def generate_industrial_data(n_rows=1000):
    np.random.seed(42)
    data = {
        'machine_id': np.random.randint(1, 11, n_rows),
        'temperature': np.random.normal(70, 10, n_rows),
        'vibration': np.random.normal(5, 2, n_rows),
        'pressure': np.random.normal(100, 15, n_rows),
        'working_hours': np.linspace(0, 5000, n_rows),
    }
    df = pd.DataFrame(data)
    # Lógica de falha: prob aumenta se temp > 85 ou vibração > 8
    df['failure'] = ((df['temperature'] > 85) | (df['vibration'] > 8)).astype(int)
    # Adicionar um pouco de ruído
    noise = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    df['failure'] = np.where(noise == 1, 1 - df['failure'], df['failure'])
    
    df.to_csv('data/industrial_maintenance_data.csv', index=False)
    print("Dados gerados com sucesso na pasta data/")

if __name__ == "__main__":
    generate_industrial_data()