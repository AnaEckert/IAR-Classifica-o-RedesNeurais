import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\nCARREGANDO E TRATANDO OS DADOS ---")

df = pd.read_csv("diabetesdataset.csv")

# Correções
colunas_com_zeros_falsos = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[colunas_com_zeros_falsos] = df[colunas_com_zeros_falsos].replace(0, np.nan)

df = df.dropna()
print(f"Quantidade de amostras restantes após a limpeza: {df.shape[0]} itens.")

coluna_alvo = "Outcome"

print("\nINICIANDO ANÁLISE EXPLORATÓRIA DE DADOS ---")

print("\nResumo Estatístico do Dataset (Sem os zeros falsos):")
print(df.describe())

print("\nVerificação de Valores Nulos/Ausentes:")
print(df.isnull().sum())

print("\nContagem de Amostras por Classe (0: Não Diabético, 1: Diabético):")
print(df[coluna_alvo].value_counts().sort_index())

# Histograma
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["Glucose"], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribuição da Glicose")
plt.xlabel("Glicose (mg/dL)")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.hist(df["BMI"], bins=20, color="salmon", edgecolor="black", alpha=0.7)
plt.title("Distribuição do IMC (BMI)")
plt.xlabel("IMC")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(12, 6))
df.drop(columns=[coluna_alvo]).boxplot(patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot dos Atributos Médicos do Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Matriz de Correlação 
plt.figure(figsize=(10, 8))

matriz_completa = df.corr()

sns.heatmap(
    matriz_completa, 
    annot=True,          
    cmap="coolwarm",     
    fmt=".2f",           
    linewidths=0.5,      
    vmin=-1, vmax=1      
)

plt.title("Mapa de Calor da Matriz de Correlação Geral")
plt.tight_layout()
plt.show()

# Scatter plot

plt.figure(figsize=(10, 7))
classes_unicas = [0, 1]
nomes_classes = {0: "Não Diabético", 1: "Diabético"}
cores = ["skyblue", "salmon"]

for classe in classes_unicas:
    grupo = df[df[coluna_alvo] == classe]
    plt.scatter(
        grupo["Glucose"],
        grupo["BMI"],
        label=nomes_classes[classe],
        alpha=0.7,
        edgecolors="k",
        color=cores[classe]
    )

plt.title("Dispersão: Glicose vs IMC por Classe")
plt.xlabel("Glicose")
plt.ylabel("IMC (BMI)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

print("\nMatriz de Correlação (Pearson) Completa:")
matriz_correlacao = df.corr()
print(matriz_correlacao[coluna_alvo].sort_values(ascending=False))

print("\nDIVISÃO DOS DADOS ---")

X = df.drop(columns=[coluna_alvo]).values
y = df[coluna_alvo].values

X_treino, X_temp, y_treino, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_teste, y_val, y_teste = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Amostras de Treino (70%):    {X_treino.shape[0]} itens")
print(f"Amostras de Validação (15%): {X_val.shape[0]} itens")
print(f"Amostras de Teste (15%):     {X_teste.shape[0]} itens")

print("\nESCALONAMENTO DOS DADOS ---")

scaler = StandardScaler()

X_treino_escalado = scaler.fit_transform(X_treino)
X_val_escalado = scaler.transform(X_val)
X_teste_escalado = scaler.transform(X_teste)

print("Dados Escalados com Sucesso!")
print(f"Média da 1ª variável no Treino Escalado:     {X_treino_escalado[:, 0].mean():.4f} (Deve ser 0)")
print(f"Desvio Padrão da 1ª variável no Treino Esc.: {X_treino_escalado[:, 0].std():.4f} (Deve ser 1)")

print("\n--- CONFIGURANDO A ARQUITETURA DA MLP (Versão Sigmoid Binária) ---")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(16, activation='relu', input_shape=(8,), kernel_regularizer=l2(0.01)),
    
    Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
    
    Dense(1, activation="sigmoid")
])

otimizador_adam = Adam(learning_rate=0.005)

model.compile(
    optimizer=otimizador_adam,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTREINAMENTO DA REDE ---")

from tensorflow.keras.callbacks import EarlyStopping

callback_parada = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

historico = model.fit(
    X_treino_escalado, y_treino,
    validation_data=(X_val_escalado, y_val),
    epochs=150,
    batch_size=16,
    callbacks=[callback_parada],
    verbose=1
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historico.history['loss'], label='Erro de Treino (Loss)')
plt.plot(historico.history['val_loss'], label='Erro de Validação (Val Loss)')
plt.title('Histórico de Erro (Loss)')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historico.history['accuracy'], label='Acurácia de Treino')
plt.plot(historico.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Histórico de Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nAVALIAÇÃO FINAL (DADOS DE TESTE) ---")

previsoes_probabilidades = model.predict(X_teste_escalado)

y_predito = (previsoes_probabilidades >= 0.5).astype(int).flatten()

acuracia_final = accuracy_score(y_teste, y_predito)
print(f"\nAcurácia Final no Conjunto de Teste: {acuracia_final * 100:.2f}%")

target_names_report = ["Não Diabético (0)", "Diabético (1)"]
labels_metricas = ["Não Diabético", "Diabético"]

print("\nRelatório de Classificação Completo:")
print(classification_report(y_teste, y_predito, target_names=target_names_report))

cm = confusion_matrix(y_teste, y_predito)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_metricas)

print("\nGerando Matriz de Confusão visual...")
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title("Matriz de Confusão - Dataset Diabetes")
plt.show()

