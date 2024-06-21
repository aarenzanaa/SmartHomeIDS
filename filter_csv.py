import pandas as pd
from datetime import datetime

# Lista de direcciones IP para filtrar
ip_list = [
    '192.168.1.134',
    '192.168.1.135',
    '192.168.1.136',
    '192.168.1.137',
    '192.168.1.138',
    '192.168.1.147',
    '192.168.1.148',
    '192.168.1.149'
]

# Nombres de las columnas
column_names = ['ts', 'uid', 'orig_h', 'orig_p', 'resp_h', 'resp_p']

def process_file(day):
    # Formatear el nombre del archivo para incluir el día
    file_name = f'2024-04-{day:02d}.csv'

    try:
        # Leer el archivo CSV con nombres de columnas
        df = pd.read_csv(file_name, names=column_names, header=None)

        # Convertir la columna 'ts' a datetime
        df['ts'] = pd.to_datetime(df['ts'])

        # Filtrar las filas donde el campo 'resp_h' esté en la lista de IPs
        filtered_df = df[df['resp_h'].isin(ip_list)]

        # Ordenar el DataFrame filtrado por la columna 'ts'
        filtered_df = filtered_df.sort_values(by='ts')

        # Añadir la columna 'malicious' con valor True
        filtered_df['malicious'] = "true"

        # Guardar el DataFrame filtrado, ordenado y con la nueva columna en un nuevo archivo CSV
        output_file_name = f'2024-04-{day:02d}_filter.csv'
        filtered_df.to_csv(output_file_name, index=False, header=True)

        print(f"Procesado completado para el archivo '{file_name}'. Guardado como '{output_file_name}'.")

    except FileNotFoundError:
        print(f"El archivo '{file_name}' no se encontró. Continuando con el siguiente día.")
    except Exception as e:
        print(f"Ocurrió un error al procesar '{file_name}': {e}")

# Recorrer todos los días del mes (1 al 31)
for day in range(15, 31):
    process_file(day)
