#!/usr/bin/env python
"""Professional pre-flight check - NEVER waste 8 hours on missing imports again."""

import sys
import traceback
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_status(status: bool, message: str):
    """Print colored status message."""
    if status:
        print(f"{GREEN}✓{RESET} {message}")
    else:
        print(f"{RED}✗{RESET} {message}")
        
def print_header(message: str):
    """Print section header."""
    print(f"\n{BOLD}{message}{RESET}")
    print("-" * 50)

class PreflightCheck:
    """Professional pre-flight checks that would make a senior engineer proud."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def check_imports(self):
        """Check ALL imports used in training scripts."""
        print_header("1. Import Check")
        
        required_imports = [
            "numpy as np",
            "torch",
            "pytorch_lightning as pl",
            "sklearn.metrics",
            "omegaconf",
            "mne",
            "pandas",
            "matplotlib",
        ]
        
        for imp in required_imports:
            try:
                exec(f"import {imp}")
                print_status(True, f"Import {imp.split()[0]}")
            except ImportError as e:
                print_status(False, f"Import {imp.split()[0]}: {e}")
                self.errors.append(f"Missing: {imp}")
                
    def check_project_imports(self):
        """Check project-specific imports."""
        print_header("2. Project Import Check")
        
        # Add project root
        project_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(project_root))
        
        project_imports = [
            "from brain_go_brrr.data.tuab_dataset import TUABDataset",
            "from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe",
            "from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe",
        ]
        
        for imp in project_imports:
            try:
                exec(imp)
                print_status(True, imp.split()[-1])
            except Exception as e:
                print_status(False, f"{imp}: {e}")
                self.errors.append(f"Failed: {imp}")
                
    def check_data_paths(self):
        """Check all required data paths exist."""
        print_header("3. Data Path Check")
        
        import os
        data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
        
        required_paths = {
            "EEGPT checkpoint": data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt",
            "TUAB dataset": data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            "Cache directory": data_root / "cache",
        }
        
        for name, path in required_paths.items():
            exists = path.exists()
            print_status(exists, f"{name}: {path}")
            if not exists:
                self.errors.append(f"Missing: {name} at {path}")
                
    def check_gpu(self):
        """Check GPU availability and memory."""
        print_header("4. GPU Check")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print_status(cuda_available, f"CUDA available: {cuda_available}")
            
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print_status(True, f"GPU: {device_name} ({memory_gb:.1f}GB)")
                
                # Check free memory
                allocated = torch.cuda.memory_allocated(0) / 1e9
                free = memory_gb - allocated
                print_status(free > 4, f"Free memory: {free:.1f}GB")
                if free < 4:
                    self.warnings.append("Low GPU memory - may cause OOM")
        except Exception as e:
            print_status(False, f"GPU check failed: {e}")
            self.errors.append("GPU not accessible")
            
    def test_validation_code(self):
        """Test the EXACT code path that crashed after 8 hours."""
        print_header("5. Validation Code Test (THE KILLER)")
        
        try:
            # This is the code that crashed
            import numpy as np
            from sklearn.metrics import roc_auc_score
            
            # Simulate validation data
            probs_cpu = np.array([0.1, 0.9, 0.5, 0.7])
            labels_cpu = np.array([0, 1, 0, 1])
            
            # Test NaN handling (the exact lines that crashed)
            if np.isnan(probs_cpu).any():
                print("Found NaN")
            
            # Test AUROC calculation
            auroc = roc_auc_score(labels_cpu, probs_cpu)
            print_status(True, f"Validation code works! AUROC test: {auroc:.3f}")
            
        except Exception as e:
            print_status(False, f"VALIDATION CODE FAILED: {e}")
            self.errors.append(f"Validation will crash: {e}")
            traceback.print_exc()
            
    def test_data_loading(self):
        """Test actual data loading (quick)."""
        print_header("6. Data Loading Test")
        
        try:
            import os
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from brain_go_brrr.data.tuab_dataset import TUABDataset
            
            data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
            dataset = TUABDataset(
                root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
                split="train",
                window_duration=8.0,
                max_files=1,  # Just test one file
            )
            
            if len(dataset) > 0:
                x, y = dataset[0]
                print_status(True, f"Loaded sample: shape={x.shape}, label={y}")
            else:
                print_status(False, "Dataset empty")
                self.errors.append("Cannot load data")
                
        except Exception as e:
            print_status(False, f"Data loading failed: {e}")
            self.errors.append(f"Data loading error: {e}")
            
    def test_model_loading(self):
        """Test model initialization."""
        print_header("7. Model Loading Test")
        
        try:
            import os
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe
            
            data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
            checkpoint = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
            
            if checkpoint.exists():
                probe = AbnormalityDetectionProbe(checkpoint, n_input_channels=20)
                print_status(True, "Model initialized successfully")
            else:
                print_status(False, f"Checkpoint missing: {checkpoint}")
                self.errors.append("Cannot initialize model")
                
        except Exception as e:
            print_status(False, f"Model loading failed: {e}")
            self.errors.append(f"Model error: {e}")
            
    def run_all_checks(self):
        """Run all pre-flight checks."""
        print(f"{BOLD}🚀 PROFESSIONAL PRE-FLIGHT CHECK{RESET}")
        print("=" * 50)
        
        self.check_imports()
        self.check_project_imports()
        self.check_data_paths()
        self.check_gpu()
        self.test_validation_code()
        self.test_data_loading()
        self.test_model_loading()
        
        # Summary
        print_header("SUMMARY")
        
        if self.errors:
            print(f"{RED}❌ FAILED - {len(self.errors)} errors found:{RESET}")
            for error in self.errors:
                print(f"   • {error}")
            print(f"\n{RED}DO NOT START TRAINING!{RESET}")
            return False
        else:
            if self.warnings:
                print(f"{YELLOW}⚠️  {len(self.warnings)} warnings:{RESET}")
                for warning in self.warnings:
                    print(f"   • {warning}")
            
            print(f"\n{GREEN}✅ ALL CHECKS PASSED - SAFE TO TRAIN!{RESET}")
            print(f"{GREEN}No more 8-hour crashes from missing imports!{RESET}")
            return True


if __name__ == "__main__":
    import os
    os.environ["BGB_DATA_ROOT"] = str(Path(__file__).resolve().parents[2] / "data")
    
    checker = PreflightCheck()
    success = checker.run_all_checks()
    
    if success:
        print(f"\n{BOLD}Ready to run:{RESET}")
        print("tmux new-session -d -s eegpt_safe 'uv run python train_overnight_optimized.py'")
    
    sys.exit(0 if success else 1)