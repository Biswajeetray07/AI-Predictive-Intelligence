import torch
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'saved_models', 'checkpoints', 'gru_checkpoint.pt')

def inspect_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        return
        
    print(f"Inspecting checkpoint: {CHECKPOINT_PATH}")
    try:
        # Load with weights_only=False since it's a checkpoint with custom dict
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        
        print("\nCheckpoint Metadata:")
        for key in checkpoint.keys():
            if key != 'model_state_dict' and key != 'optimizer_state_dict':
                print(f"  {key}: {checkpoint[key]}")
                
        if 'epoch' in checkpoint:
            print(f"\nCurrently at Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"Best Validation Loss so far: {checkpoint['val_loss']:.6f}")
            
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
