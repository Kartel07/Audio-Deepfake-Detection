# Version-5.0/CoreModel.py
import os
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
v2_dir = os.path.join(root_dir, "Version-2.0")
v3_dir = os.path.join(root_dir, "Version-3.0")

def load_custom_module(module_name, file_path):
    """Bypasses Python's namespace collision to load identically named files."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

print("🔗 Dynamically mounting V2 and V3 architectures into V5.0 Fusion Engine...")

# Load V2 (ResNet)
v2_module = load_custom_module("CoreModel_V2", os.path.join(v2_dir, "CoreModel.py"))
ResNetDeepfakeDetector = v2_module.ResNetDeepfakeDetector

# Load V3 (Transformer)
v3_module = load_custom_module("CoreModel_V3", os.path.join(v3_dir, "CoreModel.py"))
DeepfakeDetectorXLS = v3_module.DeepfakeDetectorXLS

print("✅ Version 5.0 Core: Architectures successfully mounted.")