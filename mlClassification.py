import matplotlib.pyplot as plt
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

df = pd.read_csv(
    "baseML4.txt",
    sep=r"\s+",
    decimal=",",
    comment="#",
    names=["X1", "X2", "Classe"],
)

# Correções 

df["X1"] = pd.to_numeric(df["X1"].astype(str).str.replace(",", "."), errors="coerce")
df["X2"] = pd.to_numeric(df["X2"].astype(str).str.replace(",", "."), errors="coerce")

df = df.dropna()

# Histograma

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["X1"], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribuição da Variável X1")
plt.xlabel("Valor de X1")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.hist(df["X2"], bins=30, color="salmon", edgecolor="black", alpha=0.7)
plt.title("Distribuição da Variável X2")
plt.xlabel("Valor de X2")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# Scatter Plot

plt.figure(figsize=(10, 8))

classes_unicas = sorted(df["Classe"].unique())
num_classes = len(classes_unicas)
cmap = plt.colormaps["tab20"]

for i, classe in enumerate(classes_unicas):
    grupo = df[df["Classe"] == classe]
    plt.scatter(
        grupo["X1"],
        grupo["X2"],
        label=f"Classe {classe}",
        alpha=0.7,
        edgecolors="k",
        color=cmap(i / num_classes),
    )

plt.title("Dispersão das Classes no Espaço Bidimensional")
plt.xlabel("Variável X1")
plt.ylabel("Variável X2")
plt.axhline(0, color="black", linewidth=1, linestyle="--")  
plt.axvline(0, color="black", linewidth=1, linestyle="--") 
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

print("INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS ---")

print("\n[1] Resumo Estatístico do Dataset:")
print(df.describe())

print("\n[2] Verificação de Valores Nulos/Ausentes:")
print(df.isnull().sum())

print("\n[3] Contagem de Amostras por Classe:")
print(df["Classe"].value_counts().sort_index())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df["X1"], patch_artist=True, boxprops=dict(facecolor="skyblue"))
plt.title("Boxplot da Variável X1")
plt.ylabel("Valores")
plt.grid(True, linestyle=":", alpha=0.6)

plt.subplot(1, 2, 2)
plt.boxplot(df["X2"], patch_artist=True, boxprops=dict(facecolor="salmon"))
plt.title("Boxplot da Variável X2")
plt.ylabel("Valores")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

matriz_correlacao = df[["X1", "X2"]].corr()
print("\n[4] Matriz de Correlação (Pearson):")
print(matriz_correlacao)

print("\nINICIANDO DIVISÃO DOS DADOS ---")

X = df[["X1", "X2"]].values
y = df["Classe"].values

X_treino, X_temp, y_treino, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_teste, y_val, y_teste = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Amostras de Treino (70%):    {X_treino.shape[0]} itens")
print(f"Amostras de Validação (15%): {X_val.shape[0]} itens")
print(f"Amostras de Teste (15%):     {X_teste.shape[0]} itens")

print("\nINICIANDO ESCALONAMENTO DOS DADOS ---")

scaler = StandardScaler()

X_treino_escalado = scaler.fit_transform(X_treino)

X_val_escalado = scaler.transform(X_val)
X_teste_escalado = scaler.transform(X_teste)

print("Dados Escalados com Sucesso!")
print(f"Média de X1 no Treino Escalado:     {X_treino_escalado[:, 0].mean():.4f} (Deve ser 0)")
print(f"Desvio Padrão de X1 no Treino Esc.: {X_treino_escalado[:, 0].std():.4f} (Deve ser 1)")

print("\nINICIANDO CONSTRUÇÃO DA MLP COM REGULARIZAÇÃO L2 ---")

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


y_treino_ajustado = y_treino - 1
y_val_ajustado = y_val - 1
y_teste_ajustado = y_teste - 1

model = Sequential(
    [
        Dense(2,activation="relu", input_shape=(2,),kernel_regularizer=l2(0.01),),
        Dense(2, activation="relu", kernel_regularizer=l2(0.01)),
        Dense(4, activation="softmax"),
    ]
)

print("\nResumo da Arquitetura da Rede Neural:")
model.summary()

print("\nCONFIGURANDO HIPERPARÂMETROS, OTIMIZADOR E EARLY STOPPING ---")

otimizador_adam = tf.keras.optimizers.Adam(learning_rate=0.005)

model.compile(
    optimizer=otimizador_adam,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callback_parada = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

print("\nINICIANDO O TREINAMENTO REGULARIZADO (FORWARD & BACKPROPAGATION) ---")

historico = model.fit(
    X_treino_escalado,
    y_treino_ajustado,
    validation_data=(X_val_escalado, y_val_ajustado),
    epochs=100,
    batch_size=16,
    callbacks=[callback_parada],
    verbose=1,
)

print("\nTreinamento Concluído! ---")

print("\n--- GERANDO GRÁFICO DAS CURVAS DE APRENDIZADO ---")
plt.figure(figsize=(10, 5))

plt.plot(historico.history["loss"], label="Training Loss (Com L2)", color="blue")
plt.plot(
    historico.history["val_loss"],
    label="Validation Loss (Com L2)",
    color="orange",
    linestyle="--",
)

plt.title("Novas Curvas de Aprendizado - Modelo Regularizado")
plt.xlabel("Épocas")
plt.ylabel("Loss (Erro)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

print("\nINICIANDO AVALIAÇÃO FINAL (DADOS DE TESTE) ---")

previsoes_probabilidades = model.predict(X_teste_escalado)

y_predito = np.argmax(previsoes_probabilidades, axis=1)

acuracia_final = accuracy_score(y_teste_ajustado, y_predito)
print(f"\nAcurácia Final no Conjunto de Teste: {acuracia_final * 100:.2f}%")

labels_metricas = [f"C{int(c)}" for c in classes_unicas]
target_names_report = [f"Classe {int(c)}" for c in classes_unicas]

print("\nRelatório de Classificação Completo:")
print(
    classification_report(
        y_teste_ajustado,
        y_predito,
        target_names=target_names_report,
    )
)

cm = confusion_matrix(y_teste_ajustado, y_predito)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=labels_metricas
)

print("\nGerando Matriz de Confusão visual...")
plt.figure(figsize=(12, 10))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation='vertical')
plt.title("Matriz de Confusão - Dados de Teste")
plt.show()
