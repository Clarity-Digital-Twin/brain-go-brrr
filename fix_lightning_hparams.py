#!/usr/bin/env python3
"""Fix Lightning hparams properly with TypedDict."""

from pathlib import Path
import re

def fix_hparams():
    """Fix hparams in enhanced_abnormality_detection.py."""
    file_path = Path("src/brain_go_brrr/tasks/enhanced_abnormality_detection.py")
    content = file_path.read_text()
    
    # 1. Add TypedDict import
    content = content.replace(
        "from typing import Any",
        "from typing import Any, TypedDict"
    )
    
    # 2. Add HParams TypedDict definition after logger
    logger_line = "logger = logging.getLogger(__name__)"
    if logger_line in content:
        content = content.replace(
            logger_line,
            logger_line + """


class HParams(TypedDict, total=False):
    \"\"\"Typed hyperparameters for Lightning module.\"\"\"
    learning_rate: float
    weight_decay: float
    scheduler_type: str  # "onecycle" | "cosine" | "none"
    warmup_epochs: int
    total_epochs: int
    layer_decay: float
    batch_size: int
    max_epochs: int"""
        )
    
    # 3. Add hparams type annotation to class
    content = content.replace(
        "class EnhancedAbnormalityTask(pl.LightningModule):",
        """class EnhancedAbnormalityTask(pl.LightningModule):
    hparams: HParams  # Properly typed hyperparameters
"""
    )
    
    # 4. Fix all hparams access to use .get() for safety
    replacements = [
        ("self.hparams.scheduler_type", 'self.hparams.get("scheduler_type", "none")'),
        ("self.hparams.learning_rate", 'self.hparams.get("learning_rate", 1e-3)'),
        ("self.hparams.warmup_epochs", 'self.hparams.get("warmup_epochs", 5)'),
        ("self.hparams.total_epochs", 'self.hparams.get("total_epochs", 50)'),
        ("self.hparams.weight_decay", 'self.hparams.get("weight_decay", 0.01)'),
        ("self.hparams.layer_decay", 'self.hparams.get("layer_decay", 0.75)'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # 5. Fix configure_optimizers return type
    content = content.replace(
        "def configure_optimizers(self) -> Any:",
        "def configure_optimizers(self) -> dict[str, Any] | list[Any] | Any:"
    )
    
    # Write back
    file_path.write_text(content)
    print("✅ Fixed Lightning hparams with TypedDict")

if __name__ == "__main__":
    fix_hparams()