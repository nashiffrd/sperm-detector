import h5py

# Ganti dengan path model kamu
file_path = "model_motility.h5"

with h5py.File(file_path, "r") as f:
    print("Isi root file:")
    for key in f.keys():
        print("-", key)
    
    # Biasanya ada 'model_weights' dan 'model_config'
    if "model_weights" in f:
        print("\nLayer & weights:")
        for layer_name in f['model_weights']:
            print("-", layer_name)
