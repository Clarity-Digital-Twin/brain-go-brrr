#!/usr/bin/env python3
"""Fix all type errors in the codebase to achieve 100% green baseline."""

import re
from pathlib import Path
from typing import List, Tuple

def fix_file(file_path: Path, replacements: List[Tuple[str, str]]) -> None:
    """Apply replacements to a file."""
    content = file_path.read_text()
    for old, new in replacements:
        content = content.replace(old, new)
    file_path.write_text(content)
    print(f"Fixed {file_path}")

def main():
    """Fix all type errors systematically."""
    
    # Fix missing type parameters for generic types
    fixes = {
        # Fix dict without type parameters
        "src/brain_go_brrr/api/cache.py": [
            ("def get(self, key: str) -> dict | None:", 
             "def get(self, key: str) -> dict[str, Any] | None:"),
            ("def set(self, key: str, value: dict, ttl: int = 3600) -> bool:",
             "def set(self, key: str, value: dict[str, Any], ttl: int = 3600) -> bool:"),
            ("def get_stats(self) -> dict:",
             "def get_stats(self) -> dict[str, Any]:"),
            ("def health_check(self) -> dict:",
             "def health_check(self) -> dict[str, Any]:"),
        ],
        
        # Fix list without type parameters
        "src/brain_go_brrr/preprocessing/autoreject_adapter.py": [
            ("def _create_circular_positions(self, ch_names: list) -> dict:",
             "def _create_circular_positions(self, ch_names: list[str]) -> dict[str, Any]:"),
        ],
        
        # Fix ndarray without type parameters
        "src/brain_go_brrr/preprocessing/eeg_preprocessor.py": [
            (") -> list[np.ndarray]:",
             ") -> list[npt.NDArray[np.float64]]:"),
        ],
        
        # Fix Redis type
        "src/brain_go_brrr/infra/redis/pool.py": [
            ("def get_client(self) -> Generator[redis.Redis, None, None]:",
             "def get_client(self) -> Generator[redis.Redis[str], None, None]:"),
        ],
        
        # Fix dict in edf_streaming
        "src/brain_go_brrr/data/edf_streaming.py": [
            ("def get_info(self) -> dict:",
             "def get_info(self) -> dict[str, Any]:"),
            ("def get_file_info(self) -> dict:",
             "def get_file_info(self) -> dict[str, Any]:"),
            ("-> dict:",
             "-> dict[str, Any]:"),
        ],
        
        # Fix ndarray in window_extractor
        "src/brain_go_brrr/core/window_extractor.py": [
            ("def extract(self, data: np.ndarray, sfreq: float)",
             "def extract(self, data: npt.NDArray[np.float64], sfreq: float)"),
            ("self, data: np.ndarray, sfreq: float",
             "self, data: npt.NDArray[np.float64], sfreq: float"),
            (") -> list[np.ndarray]",
             ") -> list[npt.NDArray[np.float64]]"),
            ("def is_valid(self, window: np.ndarray) -> bool:",
             "def is_valid(self, window: npt.NDArray[np.float64]) -> bool:"),
            ("self, recordings: list[np.ndarray], sfreq: float",
             "self, recordings: list[npt.NDArray[np.float64]], sfreq: float"),
        ],
        
        # Fix dependencies 
        "src/brain_go_brrr/api/dependencies.py": [
            ("job_store: dict[str, dict] = {}",
             "job_store: dict[str, dict[str, Any]] = {}"),
            ("async def get_job_store() -> dict[str, dict]:",
             "async def get_job_store() -> dict[str, dict[str, Any]]:"),
        ],
        
        # Fix visualization
        "src/brain_go_brrr/visualization/pdf_report.py": [
            ("eeg_data: npt.NDArray | None",
             "eeg_data: npt.NDArray[np.float64] | None"),
            ("eeg_data: npt.NDArray,",
             "eeg_data: npt.NDArray[np.float64],"),
            ("ax: plt.Axes",
             'ax: "plt.Axes"'),
            ("axes: list[plt.Axes] | plt.Axes",
             'axes: list["plt.Axes"] | "plt.Axes"'),
        ],
        
        # Fix hierarchical_pipeline
        "src/brain_go_brrr/services/hierarchical_pipeline.py": [
            ("async def run_parallel(self, tasks: list) -> list[Any]:",
             "async def run_parallel(self, tasks: list[Any]) -> list[Any]:"),
            ("def run_tasks(self, tasks: list) -> list[Any]:",
             "def run_tasks(self, tasks: list[Any]) -> list[Any]:"),
        ],
        
        # Fix auth
        "src/brain_go_brrr/api/auth.py": [
            ("def verify_jwt_token(token: str) -> dict:",
             "def verify_jwt_token(token: str) -> dict[str, Any]:"),
        ],
        
        # Fix snippets/maker
        "src/brain_go_brrr/core/snippets/maker.py": [
            ("eegpt_results: dict",
             "eegpt_results: dict[str, Any]"),
        ],
        
        # Fix models
        "src/brain_go_brrr/models/eegpt_architecture.py": [
            ("n_channels: list | None = None,",
             "n_channels: list[int] | None = None,"),
            ("def prepare_chan_ids(self, channel_names: list) -> Tensor:",
             "def prepare_chan_ids(self, channel_names: list[str]) -> Tensor:"),
        ],
        
        # Fix tuab_dataset
        "src/brain_go_brrr/data/tuab_dataset.py": [
            ("class TUABDataset(Dataset):",
             "class TUABDataset(Dataset[tuple[torch.Tensor, int]]):"),
            ("self._file_cache: dict[Path, np.ndarray] = {}",
             "self._file_cache: dict[Path, npt.NDArray[np.float64]] = {}"),
            ("def _load_edf_file(self, file_path: Path) -> np.ndarray:",
             "def _load_edf_file(self, file_path: Path) -> npt.NDArray[np.float64]:"),
        ],
        
        # Fix tuab_enhanced_dataset
        "src/brain_go_brrr/data/tuab_enhanced_dataset.py": [
            ("def _load_edf_file(self, file_path: Path) -> np.ndarray:",
             "def _load_edf_file(self, file_path: Path) -> npt.NDArray[np.float64]:"),
            ("-> List[Tuple[np.ndarray, int]]:",
             "-> List[Tuple[npt.NDArray[np.float64], int]]:"),
        ],
        
        # Fix models/eegpt_model
        "src/brain_go_brrr/models/eegpt_model.py": [
            ("def extract_windows(self, data: np.ndarray, sampling_rate: int)",
             "def extract_windows(self, data: npt.NDArray[np.float64], sampling_rate: int)"),
            ("self, windows: np.ndarray | torch.Tensor",
             "self, windows: npt.NDArray[np.float64] | torch.Tensor"),
            (") -> np.ndarray:",
             ") -> npt.NDArray[np.float64]:"),
            ("notch_freq: float | list | None = None,",
             "notch_freq: float | list[float] | None = None,"),
        ],
        
        # Fix training
        "src/brain_go_brrr/training/sleep_probe_trainer.py": [
            ("class SleepDataset(Dataset):",
             "class SleepDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):"),
            ("windows: list[np.ndarray],",
             "windows: list[npt.NDArray[np.float64]],"),
            (") -> tuple[float, np.ndarray]:",
             ") -> tuple[float, npt.NDArray[np.float64]]:"),
            (") -> tuple[list[np.ndarray], list[int]]:",
             ") -> tuple[list[npt.NDArray[np.float64]], list[int]]:"),
            ("-> list[np.ndarray]:",
             "-> list[npt.NDArray[np.float64]]:"),
            (") -> tuple[tuple[list, list], tuple[list, list]]:",
             ") -> tuple[tuple[list[Any], list[Any]], tuple[list[Any], list[Any]]]:"),
        ],
        
        # Fix quality controller
        "src/brain_go_brrr/core/quality/controller.py": [
            ("reject_criteria: dict | None = None,",
             "reject_criteria: dict[str, Any] | None = None,"),
            (") -> float | dict:",
             ") -> float | dict[str, Any]:"),
            (") -> dict:",
             ") -> dict[str, Any]:"),
        ],
        
        # Fix features extractor
        "src/brain_go_brrr/core/features/extractor.py": [
            ("self._cache: dict[str, np.ndarray] = {}",
             "self._cache: dict[str, npt.NDArray[np.float64]] = {}"),
            ("def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:",
             "def extract_embeddings(self, raw: mne.io.Raw) -> npt.NDArray[np.float64]:"),
            ("-> list[np.ndarray]:",
             "-> list[npt.NDArray[np.float64]]:"),
            (") -> list[np.ndarray]:",
             ") -> list[npt.NDArray[np.float64]]:"),
            ("self, windows: list[np.ndarray],",
             "self, windows: list[npt.NDArray[np.float64]],"),
        ],
        
        # Fix abnormal detector
        "src/brain_go_brrr/core/abnormal/detector.py": [
            ("def extract_features(self, data: np.ndarray,",
             "def extract_features(self, data: npt.NDArray[np.float64],"),
            ("def _extract_windows(self, raw: mne.io.Raw) -> list[np.ndarray]:",
             "def _extract_windows(self, raw: mne.io.Raw) -> list[npt.NDArray[np.float64]]:"),
            ("def _assess_window_quality(self, window: np.ndarray) -> float:",
             "def _assess_window_quality(self, window: npt.NDArray[np.float64]) -> float:"),
            ("def _predict_window(self, window: np.ndarray) -> float:",
             "def _predict_window(self, window: npt.NDArray[np.float64]) -> float:"),
        ],
        
        # Fix tasks/enhanced_abnormality_detection
        "src/brain_go_brrr/tasks/enhanced_abnormality_detection.py": [
            ("self, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray",
             "self, labels: npt.NDArray[np.float64], preds: npt.NDArray[np.float64], probs: npt.NDArray[np.float64]"),
            ("def _get_param_groups(self) -> list:",
             "def _get_param_groups(self) -> list[dict[str, Any]]:"),
        ],
        
        # Fix sleep analyzer enhanced
        "src/brain_go_brrr/core/sleep/analyzer_enhanced.py": [
            ("def _compute_fractal_dimension(self, data: npt.NDArray) -> float:",
             "def _compute_fractal_dimension(self, data: npt.NDArray[np.float64]) -> float:"),
            ("def _compute_permutation_entropy(self, data: npt.NDArray) -> float:",
             "def _compute_permutation_entropy(self, data: npt.NDArray[np.float64]) -> float:"),
        ],
    }
    
    # Apply fixes to each file
    for file_path, replacements in fixes.items():
        path = Path(file_path)
        if path.exists():
            fix_file(path, replacements)
    
    # Remove unused type: ignore comments
    files_with_unused_ignores = [
        "src/brain_go_brrr/core/config.py",
        "src/brain_go_brrr/api/routers/resources.py",
        "src/brain_go_brrr/api/routers/jobs.py",
        "src/brain_go_brrr/api/routers/health.py",
        "src/brain_go_brrr/api/routers/queue.py",
        "src/brain_go_brrr/api/routers/cache.py",
        "src/brain_go_brrr/api/routers/sleep.py",
        "src/brain_go_brrr/api/routers/eegpt.py",
        "src/brain_go_brrr/api/routers/qc.py",
        "src/brain_go_brrr/api/app.py",
        "src/brain_go_brrr/cli.py",
        "src/brain_go_brrr/data/tuab_dataset.py",
    ]
    
    for file_path in files_with_unused_ignores:
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            # Remove [misc] ignores from decorators
            content = re.sub(r'  # type: ignore\[misc\]', '', content)
            # Remove other unused ignores
            content = re.sub(r'  # type: ignore\[no-any-return\]', '', content)
            path.write_text(content)
            print(f"Removed unused ignores from {path}")
    
    print("\n✅ All type error fixes applied!")
    print("Now run: make lint && make type-check")

if __name__ == "__main__":
    main()