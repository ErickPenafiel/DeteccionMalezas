import torch

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    # Mostrar el número de GPUs disponibles
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Mostrar información detallada sobre cada GPU
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_name(i)
        print(f"GPU {i + 1}: {gpu}")
else:
    print("CUDA is not available on this system.")
