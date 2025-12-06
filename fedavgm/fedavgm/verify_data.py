
import os
import sys
from pathlib import Path

# Add the parent directory (project root) to sys.path so we can import fedavgm
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from fedavgm.dataset import imagenette

def verify():
    print("Verifying Imagenette loading...")
    # Data is expected in project_root/data
    data_dir = project_root / "data" / "imagenette2-160"
    print(f"Looking for data in: {data_dir}")
    
    if not data_dir.exists():
        print("Data directory not found!")
        # Attempt to list what's in data
        parent_data = project_root / "data"
        if parent_data.exists():
            print(f"Contents of {parent_data}:")
            for item in parent_data.iterdir():
                print(f" - {item.name}")
        else:
            print(f"{parent_data} does not exist.")
        return

    try:
        # Load data (num_classes=10, input_shape=(128,128,3))
        # Note: We updated config to 128x128, let's test that
        x_train, y_train, x_test, y_test, input_shape, num_classes = imagenette(10, (128, 128, 3), str(data_dir))
        
        print("\nSUCCESS!")
        print(f"Train shapes: X={x_train.shape}, Y={y_train.shape}")
        print(f"Test shapes:  X={x_test.shape}, Y={y_test.shape}")
        print(f"Input shape: {input_shape}")
        print(f"Num classes: {num_classes}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
