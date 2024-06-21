import os
import json
import datetime
import pytz

# Función para convertir el timestamp en formato deseado y ajustar a horario de Madrid
def convert_timestamp(ts):
    timestamp = int(ts)
    # Definir la zona horaria de Madrid
    zona_horaria_madrid = pytz.timezone('Europe/Madrid')
    # Definir una fecha de referencia
    fecha_referencia = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
    # Calcular la fecha sumando los segundos al timestamp de referencia
    fecha_convertida = fecha_referencia + datetime.timedelta(seconds=timestamp)
    # Convertir a la zona horaria de Madrid
    fecha_convertida_madrid = fecha_convertida.astimezone(zona_horaria_madrid)
    return fecha_convertida_madrid.strftime('%Y-%m-%d %H:%M:%S')

# Directorio donde se encuentran las carpetas con los archivos de registro
directorio_base = "."

# Iterar sobre las carpetas en el directorio base
for carpeta in os.listdir(directorio_base):
    carpeta_ruta = os.path.join(directorio_base, carpeta)
    if os.path.isdir(carpeta_ruta):
        # Nombre del archivo CSV combinado
        archivo_combinado = f"{carpeta}.csv"

        # Abrir el archivo combinado para escritura
        with open(archivo_combinado, "w") as combined_file:
            # Iterar sobre los archivos en la carpeta
            for filename in os.listdir(carpeta_ruta):
                if filename.startswith("tcp_") and filename.endswith(".log"):
                    with open(os.path.join(carpeta_ruta, filename), "r") as f:
                        # Crear el nombre del archivo CSV de salida
                        csv_filename = os.path.splitext(filename)[0] + ".csv"

                        # Abrir el archivo CSV para escritura
                        with open(csv_filename, "w") as csv_file:
                            # Iterar sobre las líneas del archivo de registro
                            for line in f:
                                # Convertir la línea JSON a un diccionario
                                data = json.loads(line)

                                # Convertir el timestamp a horas, minutos y segundos y ajustar a horario de Madrid
                                timestamp = convert_timestamp(data["ts"])

                                # Escribir la línea en el archivo CSV con el timestamp modificado
                                csv_file.write(f"{timestamp},{data['uid']},{data['id.orig_h']},{data['id.orig_p']},{data['id.resp_h']},{data['id.resp_p']}\n")

                    # Combinar el contenido de cada archivo CSV en el archivo combinado
                    with open(csv_filename, "r") as csv_file:
                        combined_file.write(csv_file.read())
