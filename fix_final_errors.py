#!/usr/bin/env python3
"""Fix final syntax errors and missing imports."""

from pathlib import Path
import re

def add_imports_to_file(file_path: Path, imports: list[str]) -> None:
    """Add missing imports to a file."""
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    
    # Find the import section
    lines = content.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i
    
    # Add new imports after the last import
    for imp in imports:
        if imp not in content:
            lines.insert(last_import_idx + 1, imp)
            last_import_idx += 1
    
    file_path.write_text('\n'.join(lines))
    print(f"✓ Added imports to {file_path.name}")

def fix_syntax_errors(file_path: Path, fixes: list[tuple[str, str]]) -> None:
    """Fix syntax errors in a file."""
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    for old, new in fixes:
        content = content.replace(old, new)
    file_path.write_text(content)
    print(f"✓ Fixed syntax in {file_path.name}")

def main():
    """Fix all final errors."""
    
    # 1. Add missing imports for 'Any'
    files_needing_any = [
        "src/brain_go_brrr/api/auth.py",
        "src/brain_go_brrr/api/dependencies.py",
        "src/brain_go_brrr/preprocessing/autoreject_adapter.py",
    ]
    
    for file_path in files_needing_any:
        add_imports_to_file(Path(file_path), ["from typing import Any"])
    
    # 2. Add missing imports for 'npt'
    files_needing_npt = [
        "src/brain_go_brrr/core/abnormal/detector.py",
        "src/brain_go_brrr/core/features/extractor.py",
        "src/brain_go_brrr/core/window_extractor.py",
        "src/brain_go_brrr/preprocessing/eeg_preprocessor.py",
        "src/brain_go_brrr/training/sleep_probe_trainer.py",
        "src/brain_go_brrr/visualization/pdf_report.py",
    ]
    
    for file_path in files_needing_npt:
        add_imports_to_file(Path(file_path), ["import numpy.typing as npt"])
    
    # 3. Fix syntax error in eegpt_linear_probe_robust.py
    robust_file = Path("src/brain_go_brrr/models/eegpt_linear_probe_robust.py")
    if robust_file.exists():
        content = robust_file.read_text()
        
        # Fix malformed f-string on line 129
        content = re.sub(
            r'logger\.debug\(f"Clipped input values \(count: \{self\.clip_count\.item\(\)  # type: ignore\[operator\]\}\)"\)',
            'logger.debug(f"Clipped input values (count: {self.clip_count.item()})")',  # type: ignore[operator]
            content
        )
        
        # Fix nan_count line
        content = re.sub(
            r'"nan_count": self\.nan_count\.item\(\)  # type: ignore\[operator\],',
            '"nan_count": self.nan_count.item(),  # type: ignore[operator]',
            content
        )
        
        # Fix clip_count line  
        content = re.sub(
            r'"clip_count": self\.clip_count\.item\(\)  # type: ignore\[operator\],',
            '"clip_count": self.clip_count.item(),  # type: ignore[operator]',
            content
        )
        
        # Fix fill_ calls
        content = re.sub(
            r'self\.nan_count\.fill_\(  # type: ignore\[operator\]checkpoint\["statistics"\]\["nan_count"\]\)',
            'self.nan_count.fill_(checkpoint["statistics"]["nan_count"])  # type: ignore[operator]',
            content
        )
        
        content = re.sub(
            r'self\.clip_count\.fill_\(  # type: ignore\[operator\]checkpoint\["statistics"\]\["clip_count"\]\)',
            'self.clip_count.fill_(checkpoint["statistics"]["clip_count"])  # type: ignore[operator]',
            content
        )
        
        robust_file.write_text(content)
        print("✓ Fixed models/eegpt_linear_probe_robust.py")
    
    # 4. Fix enhanced_abnormality_detection.py indentation
    task_file = Path("src/brain_go_brrr/tasks/enhanced_abnormality_detection.py")
    if task_file.exists():
        content = task_file.read_text()
        
        # Fix commented line that broke indentation
        content = content.replace(
            "# x = self.probe.adapt_channels(x)  # TODO: Implement channel adaptation\n        # Extract features with backbone",
            "        # x = self.probe.adapt_channels(x)  # TODO: Implement channel adaptation\n        # Extract features with backbone"
        )
        
        task_file.write_text(content)
        print("✓ Fixed tasks/enhanced_abnormality_detection.py")
    
    # 5. Remove unused variable in sleep analyzer
    sleep_file = Path("src/brain_go_brrr/core/sleep/analyzer.py")
    if sleep_file.exists():
        content = sleep_file.read_text()
        content = content.replace(
            "proba = sls.predict_proba()",
            "_ = sls.predict_proba()"
        )
        sleep_file.write_text(content)
        print("✓ Fixed core/sleep/analyzer.py")
    
    # 6. Simplify if-else in eegpt_model  
    model_file = Path("src/brain_go_brrr/models/eegpt_model.py")
    if model_file.exists():
        content = model_file.read_text()
        content = content.replace(
            """            if isinstance(windows, torch.Tensor):
                window = windows[i].cpu().numpy()  # Convert to numpy for extract_features
            else:
                window = windows[i]""",
            "            window = windows[i].cpu().numpy() if isinstance(windows, torch.Tensor) else windows[i]"
        )
        model_file.write_text(content)
        print("✓ Fixed models/eegpt_model.py")
    
    print("\n✅ All final errors fixed!")
    print("Run: make lint && make type-check")

if __name__ == "__main__":
    main()