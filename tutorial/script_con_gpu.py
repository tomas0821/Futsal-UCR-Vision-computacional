#!/usr/bin/env python3
"""
script_con_gpu.py - Script que utiliza GPU

Este script demuestra cómo usar GPU con TensorFlow o PyTorch.
Verifica la disponibilidad de GPU y ejecuta un cálculo simple.
"""

import os
from datetime import datetime

def main():
    print("\n" + "="*50)
    print("Script con GPU - Futsal UCR")
    print("="*50)
    print(f"Fecha/hora: {datetime.now()}")
    print("="*50 + "\n")

    # Detectar disponibilidad de GPU
    print("Verificando disponibilidad de GPU...")

    try:
        import torch
        print("✓ PyTorch detectado")

        if torch.cuda.is_available():
            print(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            device = torch.device("cuda")
        else:
            print("⚠ GPU no disponible, usando CPU")
            device = torch.device("cpu")

        # Ejemplo: operación matricial con GPU
        print("\nEjecutando operación matricial...")
        size = 1000
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)

        # Multiplicación matricial
        C = torch.mm(A, B)
        print(f"✓ Multiplicación de matrices {size}x{size} completada")
        print(f"  Resultado shape: {C.shape}")

    except ImportError:
        try:
            import tensorflow as tf
            print("✓ TensorFlow detectado")

            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✓ GPU disponible: {len(gpus)} GPU(s)")
                for gpu in gpus:
                    print(f"  - {gpu}")
            else:
                print("⚠ GPU no disponible, usando CPU")

            # Ejemplo: operación con TensorFlow
            print("\nEjecutando operación matricial...")
            size = 1000
            A = tf.random.normal([size, size])
            B = tf.random.normal([size, size])
            C = tf.matmul(A, B)
            print(f"✓ Multiplicación de matrices {size}x{size} completada")
            print(f"  Resultado shape: {C.shape}")

        except ImportError:
            print("⚠ Ni PyTorch ni TensorFlow instalados")
            print("  Para usar GPU, instala: pip install torch o pip install tensorflow")

    print(f"\n✓ Script completado: {datetime.now()}\n")

if __name__ == "__main__":
    main()
