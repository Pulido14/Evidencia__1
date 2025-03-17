import csv
import sys
from datetime import datetime
from typing import List

class DetalleNota:
    def __init__(self, nombre_servicio: str, costo_servicio: float):
        if costo_servicio <= 0:
            raise ValueError("El costo del servicio debe ser mayor que 0.")
        self.nombre_servicio = nombre_servicio
        self.costo_servicio = costo_servicio

    def __repr__(self):
        return f"DetalleNota(nombre_servicio={self.nombre_servicio}, costo_servicio={self.costo_servicio})"

class Nota:
    folio_counter = 1

    def __init__(self, cliente: str, detalles: List[DetalleNota]):
        if not detalles:
            raise ValueError("La nota debe contener al menos un servicio.")
        self.folio = Nota.folio_counter
        Nota.folio_counter += 1
        self.fecha = datetime.now()
        self.cliente = cliente
        self.detalles = detalles
        self.cancelada = False

    @property
    def monto_a_pagar(self):
        return sum(detalle.costo_servicio for detalle in self.detalles)

    def cancelar(self):
        self.cancelada = True

    def recuperar(self):
        self.cancelada = False

    def __repr__(self):
        return (f"Nota(folio={self.folio}, fecha={self.fecha.strftime('%Y-%m-%d %H:%M:%S')}, cliente={self.cliente}, "
                f"monto_a_pagar={self.monto_a_pagar}, detalles={self.detalles}, cancelada={self.cancelada})")

def guardar_estado(notas, archivo='estado.csv'):
    with open(archivo, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['folio', 'fecha', 'cliente', 'monto_a_pagar', 'detalles', 'cancelada'])
        for nota in notas:
            detalles = '|'.join([f"{d.nombre_servicio},{d.costo_servicio}" for d in nota.detalles])
            writer.writerow([nota.folio, nota.fecha.strftime('%Y-%m-%d %H:%M:%S'), nota.cliente, nota.monto_a_pagar, detalles, nota.cancelada])

def cargar_estado(archivo='estado.csv'):
    notas = []
    try:
        with open(archivo, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                detalles = [DetalleNota(*d.split(',')) for d in row['detalles'].split('|')]
                nota = Nota(row['cliente'], detalles)
                nota.folio = int(row['folio'])
                nota.fecha = datetime.strptime(row['fecha'], '%Y-%m-%d %H:%M:%S')
                nota.cancelada = row['cancelada'] == 'True'
                notas.append(nota)
            if notas:
                Nota.folio_counter = max(nota.folio for nota in notas) + 1
    except FileNotFoundError:
        print("No se encontró un estado anterior. Iniciando sin datos.")
    return notas

notas = cargar_estado()

def registrar_nota():
    cliente = input("Nombre del cliente: ")
    detalles = []
    while True:
        nombre_servicio = input("Nombre del servicio realizado: ")
        try:
            costo_servicio = float(input("Costo del servicio realizado: "))
            detalles.append(DetalleNota(nombre_servicio, costo_servicio))
        except ValueError:
            print("Costo del servicio debe ser un número mayor que 0.")
            continue
        otro = input("¿Desea agregar otro servicio? (s/n): ").lower()
        if otro != 's':
            break
    nota = Nota(cliente, detalles)
    notas.append(nota)
    print(f"Nota registrada: {nota}")

def consultas_y_reportes():
    while True:
        print("Menú principal>Consultas y reportes")
        print("1. Consulta por período")
        print("2. Consulta por folio")
        print("3. Volver al menú principal")
        opcion = input("Seleccione una opción: ")
        if opcion == "1":
            consulta_por_periodo()
        elif opcion == "2":
            consulta_por_folio()
        elif opcion == "3":
            break

def consulta_por_periodo():
    fecha_inicio = input("Fecha inicial (YYYY-MM-DD): ")
    fecha_final = input("Fecha final (YYYY-MM-DD): ")
    try:
        fecha_inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        fecha_final = datetime.strptime(fecha_final, "%Y-%m-%d")
    except ValueError:
        print("Formato de fecha inválido. Use YYYY-MM-DD.")
        return
    notas_en_periodo = [nota for nota in notas if fecha_inicio <= nota.fecha <= fecha_final and not nota.cancelada]
    if not notas_en_periodo:
        print("No hay notas emitidas para el período indicado.")
    else:
        for nota in notas_en_periodo:
            print(nota)
        exportar = input("¿Desea exportar el reporte a MsExcel? (s/n): ").lower()
        if exportar == 's':
            with open('reporte_periodo.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['folio', 'fecha', 'cliente', 'monto_a_pagar'])
                for nota in notas_en_periodo:
                    writer.writerow([nota.folio, nota.fecha.strftime('%Y-%m-%d %H:%M:%S'), nota.cliente, nota.monto_a_pagar])
            print("Reporte exportado a 'reporte_periodo.csv'.")

def consulta_por_folio():
    folio = int(input("Ingrese el folio de la nota a consultar: "))
    nota = next((nota for nota in notas if nota.folio == folio and not nota.cancelada), None)
    if nota:
        print(nota)
    else:
        print("El folio indicado no existe o corresponde a una nota cancelada.")

def cancelar_nota():
    folio = int(input("Ingrese el folio de la nota a cancelar: "))
    nota = next((nota for nota in notas if nota.folio == folio), None)
    if nota and not nota.cancelada:
        print(nota)
        confirmacion = input("¿Confirma la cancelación de esta nota? (s/n): ").lower()
        if confirmacion == 's':
            nota.cancelar()
            print("Nota cancelada.")
        else:
            print("La nota no fue cancelada.")
    else:
        print("El folio indicado no existe o corresponde a una nota cancelada.")

def recuperar_nota():
    notas_canceladas = [nota for nota in notas if nota.cancelada]
    if not notas_canceladas:
        print("No hay notas canceladas.")      
        return
    for nota in notas_canceladas:
        print(nota)
    folio = int(input("Ingrese el folio de la nota a recuperar o 0 para no recuperar ninguna: "))
    if folio == 0:
        return
    nota = next((nota for nota in notas_canceladas if nota.folio == folio), None)
    if nota:
        print(nota)
        confirmacion = input("¿Confirma la recuperación de esta nota? (s/n): ").lower()
        if confirmacion == 's':
            nota.recuperar()
            print("Nota recuperada.")
        else:
            print("La nota no fue recuperada.")
    else:
        print("El folio indicado no corresponde a una nota cancelada.")

def main():
    while True:
        print("Menú principal")
        print("1. Registrar una nota")
        print("2. Consultas y reportes")
        print("3. Cancelar una nota")
        print("4. Recuperar una nota")
        print("5. Salir")
        opcion = input("Seleccione una opción: ")
        if opcion == "1":
            registrar_nota()
        elif opcion == "2":
            consultas_y_reportes()
        elif opcion == "3":
            cancelar_nota()
        elif opcion == "4":
            recuperar_nota()
        elif opcion == "5":
            confirmacion = input("¿Confirma que desea salir? (s/n): ").lower()
            if confirmacion == 's':
                guardar_estado(notas)
                print("Estado guardado. Saliendo del sistema.")
                sys.exit()
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
