import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import Sequential
from keras.layers import LSTM, Dense
from PyPDF2 import PdfMerger
from datetime import datetime, timedelta


pdf_merger = PdfMerger()

# Cargar los 4 archivos CSV
# Crear lista de nombres de archivos desde el 15 de abril al 15 de mayo
start_date = datetime(2024, 4, 15)
end_date = datetime(2024, 5, 15)
file_names = []

current_date = start_date
while current_date <= end_date:
    file_name = current_date.strftime('%Y-%m-%d_filter.csv')
    file_names.append(file_name)
    current_date += timedelta(days=1)

print(file_names)
# file_names = ['2024-05-12_filter.csv', '2024-05-13_filter.csv', '2024-05-14_filter.csv', '2024-05-11_filter.csv']
dfs = []

for file_name in file_names:
    df = pd.read_csv(file_name)
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour
    df['minute'] = df['ts'].dt.minute
    df['orig_h'] = df['orig_h'].apply(lambda x: int(x.replace('.', '')))
    df['resp_h'] = df['resp_h'].apply(lambda x: int(x.replace('.', '')))
    dfs.append(df)

# Concatenar los DataFrames en uno solo
data = pd.concat(dfs)

# Convertir las direcciones IP en características numéricas
X_train = data[['orig_h', 'orig_p', 'resp_h', 'resp_p', 'hour', 'minute']]
y_train = data['malicious']


# Añadir el archivo del día 15 como conjunto de datos de prueba
file_names_test = ['2024-05-12_filter.csv', '2024-05-13_filter.csv', '2024-05-14_filter.csv', '2024-05-11_filter.csv','2024-05-15_filter.csv']
dfs_test = []
for file_name in file_names_test:
    #file_name_test = '2024-05-15_filter.csv'
    df_test = pd.read_csv(file_name)
    df_test['ts'] = pd.to_datetime(df_test['ts'])
    df_test['hour'] = df_test['ts'].dt.hour
    df_test['minute'] = df_test['ts'].dt.minute
    df_test['orig_h'] = df_test['orig_h'].apply(lambda x: int(x.replace('.', '')))
    df_test['resp_h'] = df_test['resp_h'].apply(lambda x: int(x.replace('.', '')))
    dfs_test.append(df_test)

data_test = pd.concat(dfs_test)
# Convertir las direcciones IP en características numéricas
X_test = data_test[['orig_h', 'orig_p', 'resp_h', 'resp_p', 'hour', 'minute']]
y_test = data_test['malicious']
# Dividir en conjunto de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar varios modelos
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Resultados de lstm
# Preparar datos para lstm
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Definir y entrenar la lstm
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

# Evaluar el rendimiento de la lstm
y_prob_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = (y_prob_lstm > 0.5).astype(int)

# Matriz de confusión
cm_lstm = confusion_matrix(y_test, y_pred_lstm)

# Classification Report
cr_lstm = classification_report(y_test, y_pred_lstm)

# ROC Curve
fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_test, y_prob_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# Imprimir estadísticas lstm por la terminal
print("lstm Results:")
print("Confusion Matrix:")
print(cm_lstm)
print(cr_lstm)
print("\n")

# Visualización en una figura separada para lstm
plt.figure(figsize=(10, 8))

# Matriz de confusión para lstm
plt.subplot(2, 2, 1)
plt.imshow(cm_lstm, interpolation='nearest', cmap=plt.cm.Blues)
# Agregar texto dentro de cada cuadro
for i in range(cm_lstm.shape[0]):
    for j in range(cm_lstm.shape[1]):
        plt.text(j, i, str(cm_lstm[i, j]), horizontalalignment="center", color="red")

plt.title('Confusion Matrix - lstm')
plt.colorbar()
plt.xlabel('Prediction')
plt.ylabel('Real')
custom_labels = ['False', 'True']
plt.xticks([0, 1], custom_labels)
plt.yticks([0, 1], custom_labels)
plt.grid(False)

# Classification Report para lstm
plt.subplot(2, 2, 2)
cr_lines_lstm = cr_lstm.split("\n")
headers_lstm = cr_lines_lstm[0].split()
data_lstm = [line.split() for line in cr_lines_lstm[2:4]]

# Asegurarse de que las etiquetas de columna tengan la misma longitud que los datos
headers_lstm.insert(0, 'Result')
colLabels_lstm = headers_lstm[:len(data_lstm[0])]
# Añadir etiquetas adicionales si es necesario para igualar el número de columnas
if len(colLabels_lstm) < len(data_lstm[0]):
    colLabels_lstm += [''] * (len(data_lstm[0]) - len(colLabels_lstm))

plt.table(cellText=data_lstm, colLabels=colLabels_lstm, loc='center')
plt.title('Classification Report - lstm')
plt.axis('off')

# ROC Curve para lstm
plt.subplot(2, 2, 3)
plt.plot(fpr_lstm, tpr_lstm, label=f'AUC = {roc_auc_lstm:.2f}')
plt.title('ROC Curve - lstm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()

# Guardar el gráfico en un archivo PDF
pdf_filename = f'stats_lstm.pdf'
plt.savefig(pdf_filename)
plt.close()  # Cerrar la figura actual para liberar memoria

# Agregar el archivo PDF al merger
pdf_merger.append(pdf_filename)

# Cargar datos desde el CSV
data = data_test
# Crear el scatter plot
plt.figure(figsize=(12, 8))

# Definir los colores y etiquetas para la leyenda
legend_elements = [
    Line2D([0], [0], marker='o', color='r', label='Malicious', linestyle='None'),
    Line2D([0], [0], marker='o', color='b', label='False Friendly', linestyle='None'),
    Line2D([0], [0], marker='o', color='orange', label='False Malicious', linestyle='None'),
    Line2D([0], [0], marker='o', color='g', label='Friendly', linestyle='None')
]

for index, row in data.iterrows():
    timestamp = row['ts']
    resp_h = row['resp_h']
    malicious = row['malicious']
    malicious_pred = y_pred_lstm[index]
    if malicious and malicious_pred:
        color = 'r'
    elif malicious and not malicious_pred:
        color = 'r'
    elif not malicious and malicious_pred:
        color = 'orange'
    elif not malicious and not malicious_pred:
        color = 'g'
    #plt.scatter(timestamp, resp_h, marker='o', color=color)
    hour = timestamp.hour + timestamp.minute / 60  # Convertir los minutos a fracción de horas
    plt.scatter(hour, resp_h, marker='o', color=color)

# Configurar los ejes
plt.xlabel('Timestamp', fontsize=16)
plt.ylabel('IP', fontsize=16)
plt.title(f'Scatter Plot of IP vs Timestamp (lstm)', fontsize=16)
plt.grid(True)

# Agregar la leyenda
plt.legend(handles=legend_elements, fontsize=16)
plt.tight_layout()

# Guardar el gráfico en un archivo PDF
pdf_filename = f'predictions_lstm.pdf'
plt.savefig(pdf_filename)
plt.close()  # Cerrar la figura actual para liberar memoria

# Agregar el archivo PDF al merger
pdf_merger.append(pdf_filename)


# Evaluar los otros modelos con los datos del día 15
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # for i in range(len(X_test)):
    #     print(X_test[i], y_test[i], y_pred[i])

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    cr = classification_report(y_test, y_pred)

    # ROC Curve
    if model_name != 'SVM':
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # Normalizar las probabilidades
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Imprimir estadísticas por terminal
    print(f"Results for {model_name}:")
    print("Confusion Matrix:")
    print(cm)
    print(cr)
    print("\n")

    # Visualización en una figura separada
    plt.figure(figsize=(10, 8))

    # Matriz de confusión
    plt.subplot(2, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Agregar texto dentro de cada cuadro
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="red")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Prediction')
    plt.ylabel('Real')
    custom_labels = ['False', 'True']
    plt.xticks([0, 1], custom_labels)
    plt.yticks([0, 1], custom_labels)
    plt.grid(False)

    # Classification Report
    plt.subplot(2, 2, 2)
    cr_lines = cr.split("\n")
    headers = cr_lines[0].split()
    data = [line.split() for line in cr_lines[2:4]]

    # Asegurarse de que las etiquetas de columna tengan la misma longitud que los datos
    headers.insert(0, 'Result')
    colLabels = headers[:len(data[0])]
    # Añadir etiquetas adicionales si es necesario para igualar el número de columnas
    if len(colLabels) < len(data[0]):
        colLabels += [''] * (len(data[0]) - len(colLabels))

    plt.table(cellText=data, colLabels=colLabels, loc='center')
    plt.title(f'Classification Report - {model_name}')
    plt.axis('off')

    # ROC Curve
    plt.subplot(2, 2, 3)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Guardar el gráfico en un archivo PDF
    pdf_filename = f'stats_{model_name[:-4]}.pdf'
    plt.savefig(pdf_filename)
    plt.close()  # Cerrar la figura actual para liberar memoria

    # Agregar el archivo PDF al merger
    pdf_merger.append(pdf_filename)

    # Cargar datos desde el CSV
    data = data_test

    # Crear el scatter plot
    plt.figure(figsize=(12, 8))

    # Definir los colores y etiquetas para la leyenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='r', label='Malicious', linestyle='None'),
        Line2D([0], [0], marker='o', color='b', label='False Friendly', linestyle='None'),
        Line2D([0], [0], marker='o', color='orange', label='False Malicious', linestyle='None'),
        Line2D([0], [0], marker='o', color='g', label='Friendly', linestyle='None')
    ]

    for index, row in data.iterrows():
        timestamp = row['ts']
        resp_h = row['resp_h']
        malicious = row['malicious']
        malicious_pred = y_pred[index]
        if malicious and malicious_pred:
            color = 'r'
        elif malicious and not malicious_pred:
            color = 'r'
        elif not malicious and malicious_pred:
            color = 'orange'
        elif not malicious and not malicious_pred:
            color = 'g'
        #plt.scatter(timestamp, resp_h, marker='o', color=color)

        hour = timestamp.hour + timestamp.minute / 60  # Convertir los minutos a fracción de horas
        plt.scatter(hour, resp_h, marker='o', color=color)


    # Configurar los ejes
    plt.xlabel('Hour of the Day', fontsize=16)
    plt.ylabel('IP', fontsize=16)
    plt.title(f'Scatter Plot of IP vs Timestamp ({model_name})', fontsize=16)
    plt.grid(True)

    # Agregar la leyenda
    plt.legend(handles=legend_elements, fontsize=16)

    plt.tight_layout()

    # Guardar el gráfico en un archivo PDF
    pdf_filename = f'predictions_{model_name[:-4]}.pdf'
    plt.savefig(pdf_filename)
    plt.close()  # Cerrar la figura actual para liberar memoria

    # Agregar el archivo PDF al merger
    pdf_merger.append(pdf_filename)


print("Scatter plots guardados en archivos PDF.")

# Guardar el archivo combinado
output_pdf = 'predicitons_scatterplots.pdf'
with open(output_pdf, 'wb') as output:
    pdf_merger.write(output)

print(f"Los archivos PDF combinados se han guardado en '{output_pdf}'.")


plt.show()

