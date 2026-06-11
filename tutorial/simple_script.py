#!/usr/bin/env python3
"""
simple_script.py - Script simple para demostración en partición serial

Este script es un ejemplo básico que se ejecuta en la partición serial
del cluster UCR.
"""

import time
from datetime import datetime

def main():
    print("\n" + "="*50)
    print("Script Python Simple - Futsal UCR")
    print("="*50)

    # Información del sistema
    import os
    print(f"PID del proceso: {os.getpid()}")
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Fecha/hora: {datetime.now()}")
    print("="*50 + "\n")

    # Ejemplo: cálculo simple
    print("Ejecutando cálculo simple...")
    result = 0
    for i in range(1, 101):
        result += i
        if i % 20 == 0:
            print(f"  Iteración {i}/100...")

    print(f"\nSuma de 1 a 100: {result}")

    # Simular procesamiento
    print("\nProcesando datos...")
    time.sleep(2)

    print("\n✓ Script completado exitosamente")
    print(f"Fecha/hora final: {datetime.now()}\n")

if __name__ == "__main__":
    main()
