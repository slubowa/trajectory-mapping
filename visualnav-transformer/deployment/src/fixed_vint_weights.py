import torch
import sys
sys.path.append("/home/lubowasimon7/transfomer_project/visualnav-transformer/train")  # ✅ Ensure vint_train is accessible

from vint_train.models.vint.vint import ViNT  # ✅ Correct import
# Load the original checkpoint
MODEL_WEIGHTS_PATH = "/home/lubowasimon7/transfomer_project/visualnav-transformer/deployment/model_weights/vint.pth"
checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")

# Check if the model instance is stored instead of state_dict
if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
    print("Extracting state_dict from full model checkpoint...")
    model_state_dict = checkpoint["model"].state_dict()
    torch.save(model_state_dict, "/home/lubowasimon7/transfomer_project/visualnav-transformer/deployment/model_weights/vint_fixed.pth")
    print("Saved fixed model weights as vint_fixed.pth")
else:
    print("Checkpoint already contains a valid state_dict.")
