import os

def filter_labels(data_dir):
    label_dirs = [
        os.path.join(data_dir, 'train/labels'),
        os.path.join(data_dir, 'valid/labels'),
        os.path.join(data_dir, 'test/labels')
    ]
    
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue
            
        print(f"Filtrando etiquetas en: {label_dir}")
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(label_dir, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.split()
                    class_id = int(parts[0])
                    if class_id == 0:
                        continue # Ignorar el balón
                    
                    # Decrementar el resto de IDs
                    new_class_id = class_id - 1
                    new_lines.append(f"{new_class_id} " + " ".join(parts[1:]) + "\n")
                
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)

if __name__ == "__main__":
    dataset_path = "futsal-ucr-8"
    filter_labels(dataset_path)
