# Failure Mode Analysis for Brain-Go-Brrr

## Executive Summary

This document provides a comprehensive analysis of potential failure modes in the Brain-Go-Brrr EEG analysis pipeline, along with detection strategies, mitigation approaches, and recovery procedures. Given the medical nature of the application, we prioritize patient safety and result reliability.

## Critical Failure Categories

### Severity Levels
- **S1 (Critical)**: Could lead to patient harm or missed critical diagnoses
- **S2 (Major)**: Significant impact on accuracy or system availability
- **S3 (Minor)**: Degraded performance or user experience
- **S4 (Cosmetic)**: Minimal impact on functionality

## 1. Data Input Failures

### 1.1 Corrupted EDF Files (S1)
```python
# src/brain_go_brrr/failures/data_validation.py
from typing import Optional, Dict, List
import mne
import numpy as np
from pathlib import Path

class EDFValidator:
    """Validate EDF files before processing."""
    
    def __init__(self):
        self.min_duration = 60  # seconds
        self.min_channels = 19
        self.valid_sfreqs = [100, 128, 200, 250, 256, 500, 512, 1000]
        
    def validate_edf(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive EDF validation."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Attempt to read header only first
            with open(file_path, 'rb') as f:
                header = f.read(256)
                if not header.startswith(b'0       '):
                    results["valid"] = False
                    results["errors"].append("Invalid EDF header format")
                    return results
            
            # Try to load the file
            raw = mne.io.read_raw_edf(
                file_path,
                preload=False,
                stim_channel='auto',
                verbose='ERROR'
            )
            
            # Check duration
            duration = raw.n_times / raw.info['sfreq']
            results["metadata"]["duration"] = duration
            if duration < self.min_duration:
                results["errors"].append(
                    f"Recording too short: {duration:.1f}s < {self.min_duration}s"
                )
                results["valid"] = False
            
            # Check channels
            n_channels = len(raw.ch_names)
            results["metadata"]["n_channels"] = n_channels
            if n_channels < self.min_channels:
                results["errors"].append(
                    f"Insufficient channels: {n_channels} < {self.min_channels}"
                )
                results["valid"] = False
            
            # Check sampling rate
            sfreq = raw.info['sfreq']
            results["metadata"]["sfreq"] = sfreq
            if sfreq not in self.valid_sfreqs:
                # Will resample, just warn
                results["warnings"].append(
                    f"Non-standard sampling rate: {sfreq}Hz"
                )
            
            # Check for NaN or infinite values
            sample_data = raw.get_data(start=0, stop=int(sfreq))
            if np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data)):
                results["errors"].append("Data contains NaN or infinite values")
                results["valid"] = False
            
            # Check for flat channels
            flat_channels = []
            for i, ch_name in enumerate(raw.ch_names):
                if np.std(sample_data[i]) < 1e-10:
                    flat_channels.append(ch_name)
            
            if flat_channels:
                results["warnings"].append(
                    f"Flat channels detected: {', '.join(flat_channels)}"
                )
            
            # Check for extreme values
            max_amplitude = np.max(np.abs(sample_data))
            if max_amplitude > 1e-3:  # > 1mV
                results["warnings"].append(
                    f"Extreme amplitudes detected: {max_amplitude*1e6:.1f}µV"
                )
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to read EDF: {str(e)}")
        
        return results


class DataIntegrityChecker:
    """Check data integrity throughout pipeline."""
    
    @staticmethod
    def validate_window(window: np.ndarray, expected_shape: Tuple[int, int]) -> bool:
        """Validate single window of data."""
        # Check shape
        if window.shape != expected_shape:
            return False
        
        # Check for NaN
        if np.any(np.isnan(window)):
            return False
        
        # Check for reasonable values (EEG should be in µV range)
        if np.max(np.abs(window)) > 1000e-6:  # > 1000 µV
            return False
        
        # Check for DC offset
        if np.abs(np.mean(window)) > 100e-6:  # > 100 µV DC
            return False
        
        return True
```

### 1.2 Channel Mismatch (S2)
```python
class ChannelMismatchHandler:
    """Handle channel naming and ordering mismatches."""
    
    STANDARD_CHANNELS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2'
    ]
    
    # Alternative naming conventions
    CHANNEL_ALIASES = {
        'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
        'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
        'A1': 'M1', 'A2': 'M2'
    }
    
    def standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Standardize channel names and select required channels."""
        # First, rename aliases
        rename_mapping = {}
        for old_name in raw.ch_names:
            if old_name.upper() in self.CHANNEL_ALIASES:
                new_name = self.CHANNEL_ALIASES[old_name.upper()]
                rename_mapping[old_name] = new_name
        
        if rename_mapping:
            raw.rename_channels(rename_mapping)
        
        # Find available standard channels
        available_standard = []
        missing_channels = []
        
        for ch in self.STANDARD_CHANNELS:
            if ch in raw.ch_names:
                available_standard.append(ch)
            else:
                # Try case-insensitive match
                found = False
                for raw_ch in raw.ch_names:
                    if raw_ch.upper() == ch.upper():
                        rename_mapping = {raw_ch: ch}
                        raw.rename_channels(rename_mapping)
                        available_standard.append(ch)
                        found = True
                        break
                
                if not found:
                    missing_channels.append(ch)
        
        # Check if we have minimum channels
        if len(available_standard) < 15:
            raise ValueError(
                f"Insufficient standard channels: {len(available_standard)}/19. "
                f"Missing: {', '.join(missing_channels)}"
            )
        
        # Select and reorder channels
        raw.pick_channels(available_standard, ordered=True)
        
        # Log warning about missing channels
        if missing_channels:
            logger.warning(
                f"Missing channels will be interpolated: {', '.join(missing_channels)}"
            )
        
        return raw
```

## 2. Model Loading Failures

### 2.1 Checkpoint Corruption (S1)
```python
# src/brain_go_brrr/failures/model_validation.py
import torch
import hashlib
from typing import Optional

class ModelValidator:
    """Validate model checkpoints before use."""
    
    EXPECTED_CHECKSUMS = {
        "eegpt_mcae_58chs_4s_large4E.ckpt": "a1b2c3d4e5f6...",  # SHA256
        "linear_probe_tuab.pt": "f6e5d4c3b2a1..."
    }
    
    @classmethod
    def validate_checkpoint(
        cls,
        checkpoint_path: Path,
        expected_checksum: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate model checkpoint integrity."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        # Check file exists
        if not checkpoint_path.exists():
            results["valid"] = False
            results["errors"].append(f"Checkpoint not found: {checkpoint_path}")
            return results
        
        # Check file size
        file_size = checkpoint_path.stat().st_size
        results["metadata"]["file_size_mb"] = file_size / 1024 / 1024
        
        if file_size < 1024 * 1024:  # < 1MB
            results["valid"] = False
            results["errors"].append("Checkpoint file too small")
            return results
        
        # Verify checksum if provided
        if expected_checksum or checkpoint_path.name in cls.EXPECTED_CHECKSUMS:
            expected = expected_checksum or cls.EXPECTED_CHECKSUMS[checkpoint_path.name]
            actual = cls._calculate_checksum(checkpoint_path)
            
            if actual != expected:
                results["valid"] = False
                results["errors"].append(
                    f"Checksum mismatch. Expected: {expected}, Got: {actual}"
                )
                return results
        
        # Try to load checkpoint
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=True
            )
            
            # Validate checkpoint structure
            if 'state_dict' not in checkpoint:
                results["errors"].append("Missing 'state_dict' in checkpoint")
                results["valid"] = False
            
            if 'config' in checkpoint:
                results["metadata"]["config"] = checkpoint['config']
            
            # Check for required keys in state_dict
            state_dict = checkpoint.get('state_dict', {})
            required_keys = [
                'patch_embed.proj.weight',
                'pos_embed',
                'cls_token'
            ]
            
            for key in required_keys:
                if not any(k.startswith(key) for k in state_dict.keys()):
                    results["errors"].append(f"Missing required key: {key}")
                    results["valid"] = False
            
            # Check tensor integrity
            for key, tensor in state_dict.items():
                if torch.isnan(tensor).any():
                    results["errors"].append(f"NaN values in {key}")
                    results["valid"] = False
                
                if torch.isinf(tensor).any():
                    results["errors"].append(f"Inf values in {key}")
                    results["valid"] = False
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to load checkpoint: {str(e)}")
        
        return results
    
    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
```

### 2.2 Model Architecture Mismatch (S2)
```python
class ModelArchitectureValidator:
    """Ensure model architecture matches expectations."""
    
    @staticmethod
    def validate_eegpt_architecture(model: nn.Module) -> bool:
        """Validate EEGPT model architecture."""
        expected_layers = {
            'patch_embed': nn.Conv2d,
            'pos_embed': nn.Parameter,
            'blocks': nn.ModuleList,
            'norm': nn.LayerNorm
        }
        
        for name, expected_type in expected_layers.items():
            if not hasattr(model, name):
                logger.error(f"Missing layer: {name}")
                return False
            
            layer = getattr(model, name)
            if not isinstance(layer, expected_type):
                logger.error(
                    f"Layer {name} has wrong type. "
                    f"Expected {expected_type}, got {type(layer)}"
                )
                return False
        
        # Check dimensions
        if model.pos_embed.shape[1] != 1 + (256 * 8) // 64:  # 1 + num_patches
            logger.error("Positional embedding has wrong shape")
            return False
        
        return True
```

## 3. Processing Failures

### 3.1 Memory Exhaustion (S2)
```python
# src/brain_go_brrr/failures/memory_management.py
import psutil
import gc
from functools import wraps

class MemoryGuard:
    """Prevent memory exhaustion during processing."""
    
    def __init__(
        self,
        max_memory_percent: float = 90.0,
        emergency_threshold_gb: float = 1.0
    ):
        self.max_memory_percent = max_memory_percent
        self.emergency_threshold_gb = emergency_threshold_gb
        
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        
        return {
            "used_gb": memory.used / 1024**3,
            "available_gb": memory.available / 1024**3,
            "percent": memory.percent,
            "total_gb": memory.total / 1024**3
        }
    
    def ensure_memory_available(self, required_gb: float) -> bool:
        """Ensure enough memory is available for operation."""
        mem_info = self.check_memory()
        
        if mem_info["available_gb"] < required_gb:
            # Try garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Check again
            mem_info = self.check_memory()
            if mem_info["available_gb"] < required_gb:
                raise MemoryError(
                    f"Insufficient memory. Required: {required_gb:.1f}GB, "
                    f"Available: {mem_info['available_gb']:.1f}GB"
                )
        
        return True
    
    def monitor_memory(self, func):
        """Decorator to monitor memory usage during function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before
            before = self.check_memory()
            
            if before["percent"] > self.max_memory_percent:
                raise MemoryError(
                    f"Memory usage too high: {before['percent']:.1f}%"
                )
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Check memory after
                after = self.check_memory()
                memory_increase = after["used_gb"] - before["used_gb"]
                
                if memory_increase > 2.0:  # More than 2GB increase
                    logger.warning(
                        f"Large memory increase in {func.__name__}: "
                        f"{memory_increase:.1f}GB"
                    )
                
                # Emergency cleanup if needed
                if after["available_gb"] < self.emergency_threshold_gb:
                    logger.warning("Emergency memory cleanup triggered")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return result
        
        return wrapper


class BatchSizeAdapter:
    """Dynamically adjust batch size based on memory."""
    
    def __init__(self, initial_batch_size: int = 32):
        self.batch_size = initial_batch_size
        self.min_batch_size = 1
        self.memory_guard = MemoryGuard()
        
    def adapt_batch_size(self, memory_error: bool = False) -> int:
        """Adapt batch size based on memory availability."""
        if memory_error:
            # Reduce batch size
            self.batch_size = max(self.min_batch_size, self.batch_size // 2)
            logger.warning(f"Reduced batch size to {self.batch_size}")
        else:
            # Check if we can increase
            mem_info = self.memory_guard.check_memory()
            if mem_info["percent"] < 50 and self.batch_size < 64:
                self.batch_size = min(64, self.batch_size * 2)
                logger.info(f"Increased batch size to {self.batch_size}")
        
        return self.batch_size
```

### 3.2 NaN/Inf Propagation (S1)
```python
class NumericalStabilityGuard:
    """Prevent numerical instability in processing pipeline."""
    
    @staticmethod
    def check_tensor_validity(
        tensor: torch.Tensor,
        name: str = "tensor"
    ) -> bool:
        """Check tensor for NaN/Inf values."""
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN detected in {name}")
        
        if torch.isinf(tensor).any():
            raise ValueError(f"Inf detected in {name}")
        
        # Check for extreme values
        max_val = torch.max(torch.abs(tensor)).item()
        if max_val > 1e10:
            logger.warning(
                f"Extreme values in {name}: max abs = {max_val:.2e}"
            )
        
        return True
    
    @staticmethod
    def safe_normalize(
        tensor: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Safe normalization preventing NaN."""
        # Calculate norm with epsilon
        norm = torch.norm(tensor, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        
        # Normalize
        normalized = tensor / norm
        
        # Final check
        if torch.isnan(normalized).any():
            logger.error("NaN after normalization, returning zeros")
            return torch.zeros_like(tensor)
        
        return normalized
    
    @staticmethod
    def gradient_check_hook(grad: torch.Tensor) -> Optional[torch.Tensor]:
        """Hook to check gradients during backprop."""
        if grad is None:
            return None
        
        # Check for NaN/Inf
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            logger.error("NaN/Inf in gradients, clipping to zero")
            return torch.zeros_like(grad)
        
        # Clip extreme gradients
        max_grad = 10.0
        if torch.max(torch.abs(grad)) > max_grad:
            return torch.clamp(grad, -max_grad, max_grad)
        
        return grad
```

## 4. Result Validation Failures

### 4.1 Impossible Results (S1)
```python
# src/brain_go_brrr/failures/result_validation.py
class ResultValidator:
    """Validate analysis results for clinical plausibility."""
    
    def validate_abnormality_result(
        self,
        result: AbnormalityResult
    ) -> Dict[str, Any]:
        """Validate abnormality detection results."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check probability bounds
        if not (0 <= result.abnormal_probability <= 1):
            validation["valid"] = False
            validation["errors"].append(
                f"Invalid probability: {result.abnormal_probability}"
            )
        
        # Check confidence consistency
        confidence = result.confidence
        if result.is_abnormal and confidence < 0.5:
            validation["warnings"].append(
                "Low confidence for abnormal classification"
            )
        
        # Check triage priority consistency
        if result.triage_priority == "urgent" and confidence < 0.8:
            validation["warnings"].append(
                "Urgent triage with confidence < 0.8"
            )
        
        # Validate feature importance if present
        if hasattr(result, 'feature_importance'):
            if not np.isclose(np.sum(result.feature_importance), 1.0, atol=0.01):
                validation["warnings"].append(
                    "Feature importance doesn't sum to 1"
                )
        
        return validation
    
    def validate_sleep_stages(
        self,
        hypnogram: np.ndarray,
        duration_hours: float
    ) -> Dict[str, Any]:
        """Validate sleep staging results."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check stage values
        valid_stages = {0, 1, 2, 3, 4}  # W, N1, N2, N3, REM
        unique_stages = set(np.unique(hypnogram))
        
        if not unique_stages.issubset(valid_stages):
            validation["valid"] = False
            validation["errors"].append(
                f"Invalid sleep stages: {unique_stages - valid_stages}"
            )
        
        # Check stage transitions
        transitions = np.diff(hypnogram)
        
        # Unusual transitions (e.g., W -> N3)
        for i in range(len(transitions)):
            if hypnogram[i] == 0 and hypnogram[i+1] == 3:  # W -> N3
                validation["warnings"].append(
                    f"Unusual transition W->N3 at epoch {i}"
                )
        
        # Check stage percentages
        stage_counts = np.bincount(hypnogram, minlength=5)
        stage_percentages = stage_counts / len(hypnogram) * 100
        
        # Validate against typical ranges
        typical_ranges = {
            0: (5, 25),    # Wake: 5-25%
            1: (2, 10),    # N1: 2-10%
            2: (40, 60),   # N2: 40-60%
            3: (10, 25),   # N3: 10-25%
            4: (15, 30)    # REM: 15-30%
        }
        
        for stage, (min_pct, max_pct) in typical_ranges.items():
            pct = stage_percentages[stage]
            if pct < min_pct or pct > max_pct:
                validation["warnings"].append(
                    f"Unusual {['W', 'N1', 'N2', 'N3', 'REM'][stage]} "
                    f"percentage: {pct:.1f}%"
                )
        
        return validation
```

### 4.2 Confidence Calibration (S2)
```python
class ConfidenceCalibrator:
    """Ensure model confidence scores are well-calibrated."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_map = None
        
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ):
        """Fit calibration mapping."""
        # Calculate calibration error
        ece = self._expected_calibration_error(probabilities, labels)
        
        if ece > 0.1:  # ECE > 10%
            logger.warning(f"Poor calibration detected: ECE = {ece:.3f}")
            
            # Fit isotonic regression for calibration
            from sklearn.isotonic import IsotonicRegression
            self.calibration_map = IsotonicRegression(
                out_of_bounds='clip'
            ).fit(probabilities, labels)
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if self.calibration_map is None:
            return probabilities
        
        return self.calibration_map.transform(probabilities)
    
    def _expected_calibration_error(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0
        
        for i in range(self.n_bins):
            # Get bin mask
            bin_mask = (
                (probabilities > bin_boundaries[i]) &
                (probabilities <= bin_boundaries[i + 1])
            )
            
            if np.sum(bin_mask) > 0:
                # Calculate accuracy and confidence in bin
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(probabilities[bin_mask])
                bin_weight = np.sum(bin_mask) / len(probabilities)
                
                # Add to ECE
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
```

## 5. System-Level Failures

### 5.1 Database Connection Loss (S2)
```python
# src/brain_go_brrr/failures/system_resilience.py
import asyncio
from typing import Callable
import asyncpg

class ResilientDatabaseConnection:
    """Database connection with automatic retry and failover."""
    
    def __init__(
        self,
        primary_dsn: str,
        replica_dsns: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.primary_dsn = primary_dsn
        self.replica_dsns = replica_dsns
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.current_connection = None
        self.is_readonly = False
        
    async def execute_with_retry(
        self,
        query: str,
        *args,
        readonly: bool = False
    ) -> Any:
        """Execute query with automatic retry and failover."""
        last_error = None
        
        # Try primary first (unless readonly)
        if not readonly and not self.is_readonly:
            for attempt in range(self.max_retries):
                try:
                    if not self.current_connection:
                        self.current_connection = await asyncpg.connect(
                            self.primary_dsn
                        )
                    
                    result = await self.current_connection.fetch(query, *args)
                    return result
                    
                except (asyncpg.PostgresConnectionError, OSError) as e:
                    last_error = e
                    logger.warning(
                        f"Primary DB connection failed (attempt {attempt + 1}): {e}"
                    )
                    self.current_connection = None
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # Failover to replicas
        for replica_dsn in self.replica_dsns:
            for attempt in range(self.max_retries):
                try:
                    replica_conn = await asyncpg.connect(replica_dsn)
                    result = await replica_conn.fetch(query, *args)
                    
                    # Switch to readonly mode
                    self.is_readonly = True
                    logger.warning("Switched to readonly replica")
                    
                    return result
                    
                except (asyncpg.PostgresConnectionError, OSError) as e:
                    last_error = e
                    await asyncio.sleep(self.retry_delay)
        
        # All attempts failed
        raise DatabaseUnavailableError(
            f"All database connections failed. Last error: {last_error}"
        )
```

### 5.2 Queue Overflow (S2)
```python
class ResilientQueue:
    """Queue with overflow protection and priority handling."""
    
    def __init__(
        self,
        redis_url: str,
        max_queue_size: int = 10000,
        overflow_strategy: str = "reject"  # reject, spill_to_disk, drop_oldest
    ):
        self.redis_url = redis_url
        self.max_queue_size = max_queue_size
        self.overflow_strategy = overflow_strategy
        self.redis = None
        
    async def submit_job(
        self,
        job_id: str,
        job_data: Dict,
        priority: str = "normal"
    ) -> bool:
        """Submit job with overflow protection."""
        # Check queue size
        queue_size = await self._get_queue_size()
        
        if queue_size >= self.max_queue_size:
            return await self._handle_overflow(job_id, job_data, priority)
        
        # Calculate priority score
        priority_scores = {"urgent": 0, "expedite": 1, "normal": 2}
        score = priority_scores.get(priority, 2)
        
        # Add to sorted set
        await self.redis.zadd(
            "analysis_queue",
            {job_id: score}
        )
        
        # Store job data
        await self.redis.hset(
            f"job:{job_id}",
            mapping=job_data
        )
        
        return True
    
    async def _handle_overflow(
        self,
        job_id: str,
        job_data: Dict,
        priority: str
    ) -> bool:
        """Handle queue overflow based on strategy."""
        if self.overflow_strategy == "reject":
            logger.error(f"Queue full, rejecting job {job_id}")
            return False
            
        elif self.overflow_strategy == "spill_to_disk":
            # Save to disk queue
            disk_queue_path = Path("/tmp/queue_overflow")
            disk_queue_path.mkdir(exist_ok=True)
            
            with open(disk_queue_path / f"{job_id}.json", "w") as f:
                json.dump(job_data, f)
            
            logger.warning(f"Queue full, spilled {job_id} to disk")
            return True
            
        elif self.overflow_strategy == "drop_oldest":
            # Remove lowest priority job
            oldest = await self.redis.zrange("analysis_queue", 0, 0)
            if oldest:
                await self.redis.zrem("analysis_queue", oldest[0])
                await self.redis.delete(f"job:{oldest[0]}")
                logger.warning(f"Dropped oldest job {oldest[0]} to make room")
            
            # Try again
            return await self.submit_job(job_id, job_data, priority)
        
        return False
```

## 6. Recovery Procedures

### 6.1 Automated Recovery
```python
# src/brain_go_brrr/failures/recovery.py
class AutomatedRecoverySystem:
    """Automated recovery from common failures."""
    
    def __init__(self):
        self.recovery_strategies = {
            MemoryError: self._recover_from_memory_error,
            RuntimeError: self._recover_from_runtime_error,
            ConnectionError: self._recover_from_connection_error,
            ValueError: self._recover_from_value_error
        }
        self.max_recovery_attempts = 3
        
    async def execute_with_recovery(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with automatic recovery."""
        last_error = None
        
        for attempt in range(self.max_recovery_attempts):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}"
                )
                
                # Try recovery strategy
                recovery_func = self.recovery_strategies.get(
                    type(e),
                    self._generic_recovery
                )
                
                recovered = await recovery_func(e, func, args, kwargs)
                if not recovered:
                    break
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        # All attempts failed
        raise RecoveryFailedError(
            f"Failed after {self.max_recovery_attempts} attempts"
        ) from last_error
    
    async def _recover_from_memory_error(
        self,
        error: MemoryError,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> bool:
        """Recover from memory errors."""
        logger.info("Attempting memory error recovery")
        
        # Clear caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size if applicable
        if 'batch_size' in kwargs:
            kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
            logger.info(f"Reduced batch size to {kwargs['batch_size']}")
        
        return True
    
    async def _recover_from_connection_error(
        self,
        error: ConnectionError,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> bool:
        """Recover from connection errors."""
        logger.info("Attempting connection error recovery")
        
        # Reset connections
        if hasattr(self, 'connection_pool'):
            await self.connection_pool.clear()
        
        # Wait for services to recover
        await asyncio.sleep(5)
        
        return True
```

### 6.2 Manual Recovery Procedures
```python
class ManualRecoveryGuide:
    """Guide for manual recovery procedures."""
    
    RECOVERY_PROCEDURES = {
        "corrupted_cache": """
        # Corrupted Cache Recovery
        1. Stop all running processes
        2. Clear cache directory:
           rm -rf /data/cache/tuab_enhanced/*
        3. Rebuild cache:
           python scripts/build_tuab_cache.py
        4. Verify cache integrity:
           python scripts/validate_cache.py
        5. Restart services
        """,
        
        "model_loading_failure": """
        # Model Loading Failure Recovery
        1. Verify checkpoint integrity:
           python scripts/validate_checkpoint.py --path /data/models/eegpt.ckpt
        2. If corrupted, re-download:
           python scripts/download_models.py --model eegpt
        3. Verify SHA256 checksum matches expected
        4. Test model loading:
           python scripts/test_model_loading.py
        5. Clear model cache and restart
        """,
        
        "database_corruption": """
        # Database Corruption Recovery
        1. Stop application servers
        2. Create backup of current state:
           pg_dump -h localhost -U postgres braindb > backup_$(date +%Y%m%d).sql
        3. Check database integrity:
           psql -U postgres -d braindb -c "VACUUM ANALYZE;"
        4. If corruption detected, restore from backup:
           psql -U postgres -d braindb < last_known_good_backup.sql
        5. Replay transaction logs if available
        6. Verify data integrity
        7. Restart application servers
        """,
        
        "queue_deadlock": """
        # Queue Deadlock Recovery
        1. Identify stuck jobs:
           redis-cli ZRANGE analysis_queue 0 -1 WITHSCORES
        2. Check job states:
           for job in $(redis-cli ZRANGE analysis_queue 0 -1); do
               redis-cli HGETALL job:$job
           done
        3. Clear stuck jobs:
           redis-cli DEL analysis_queue
        4. Resubmit jobs from backup queue
        5. Monitor queue health
        """
    }
    
    @classmethod
    def get_recovery_procedure(cls, failure_type: str) -> str:
        """Get recovery procedure for failure type."""
        return cls.RECOVERY_PROCEDURES.get(
            failure_type,
            "No specific recovery procedure available. Contact support."
        )
```

## 7. Monitoring and Alerting

### 7.1 Health Checks
```python
# src/brain_go_brrr/failures/health_checks.py
class SystemHealthMonitor:
    """Monitor system health and alert on issues."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.health_checks = [
            self.check_model_availability,
            self.check_database_connectivity,
            self.check_queue_health,
            self.check_memory_usage,
            self.check_disk_space,
            self.check_gpu_health
        ]
        
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for check in self.health_checks:
            check_name = check.__name__
            try:
                result = await check()
                results[check_name] = result
                
                if not result["healthy"]:
                    overall_healthy = False
                    await self.alert_manager.send_alert(
                        severity=result.get("severity", "warning"),
                        message=f"{check_name}: {result['message']}"
                    )
                    
            except Exception as e:
                results[check_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        results["overall_healthy"] = overall_healthy
        results["timestamp"] = datetime.utcnow().isoformat()
        
        return results
    
    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if models are loaded and responsive."""
        try:
            # Test EEGPT
            test_input = torch.randn(1, 20, 2048)
            output = self.eegpt_model(test_input)
            
            return {
                "healthy": True,
                "response_time_ms": 50,
                "model_version": "1.0.0"
            }
        except Exception as e:
            return {
                "healthy": False,
                "severity": "critical",
                "message": f"Model unavailable: {e}"
            }
```

### 7.2 Failure Metrics
```python
class FailureMetricsCollector:
    """Collect and analyze failure metrics."""
    
    def __init__(self):
        self.failure_counts = defaultdict(int)
        self.recovery_success = defaultdict(int)
        self.recovery_failures = defaultdict(int)
        
    def record_failure(
        self,
        failure_type: str,
        severity: str,
        context: Dict[str, Any]
    ):
        """Record failure occurrence."""
        self.failure_counts[failure_type] += 1
        
        # Send to monitoring system
        metrics.failure_counter.labels(
            failure_type=failure_type,
            severity=severity
        ).inc()
        
        # Log for analysis
        logger.error(
            f"Failure recorded: {failure_type}",
            extra={
                "failure_type": failure_type,
                "severity": severity,
                "context": context
            }
        )
    
    def record_recovery(
        self,
        failure_type: str,
        success: bool,
        recovery_time_s: float
    ):
        """Record recovery attempt."""
        if success:
            self.recovery_success[failure_type] += 1
            metrics.recovery_success.labels(
                failure_type=failure_type
            ).inc()
        else:
            self.recovery_failures[failure_type] += 1
            metrics.recovery_failure.labels(
                failure_type=failure_type
            ).inc()
        
        metrics.recovery_duration.labels(
            failure_type=failure_type
        ).observe(recovery_time_s)
    
    def get_failure_report(self) -> Dict[str, Any]:
        """Generate failure analysis report."""
        total_failures = sum(self.failure_counts.values())
        total_recoveries = sum(self.recovery_success.values())
        
        return {
            "total_failures": total_failures,
            "total_successful_recoveries": total_recoveries,
            "recovery_rate": total_recoveries / max(total_failures, 1),
            "top_failures": dict(
                sorted(
                    self.failure_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "recovery_success_rates": {
                failure_type: self.recovery_success[failure_type] /
                max(self.recovery_success[failure_type] +
                    self.recovery_failures[failure_type], 1)
                for failure_type in self.failure_counts.keys()
            }
        }
```

## Conclusion

This comprehensive failure mode analysis ensures Brain-Go-Brrr can handle various failure scenarios gracefully, maintaining patient safety and system reliability. Regular testing of these failure scenarios and recovery procedures is essential for maintaining a robust production system.