# Hyperparameter Optimization Guide for Brain-Go-Brrr

## Executive Summary

This guide provides comprehensive hyperparameter optimization strategies for the Brain-Go-Brrr EEG analysis pipeline. We define systematic approaches for tuning model performance across EEGPT, AutoReject, abnormality detection, and sleep staging components.

## Optimization Strategy Overview

### Key Principles
1. **Bayesian Optimization**: More efficient than grid search for high-dimensional spaces
2. **Multi-Objective**: Balance accuracy, speed, and memory usage
3. **Cross-Validation**: Ensure robust performance across different EEG recordings
4. **Early Stopping**: Save computation on unpromising configurations
5. **Reproducibility**: Fix random seeds and log all experiments

## Component-Specific Hyperparameters

### 1. EEGPT Model Tuning

#### Core Hyperparameters
```python
# src/brain_go_brrr/optimization/eegpt_tuning.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import optuna
from optuna.integration import PyTorchLightningPruningCallback

@dataclass
class EEGPTHyperparameters:
    """EEGPT hyperparameter configuration."""
    
    # Model Architecture
    patch_size: int = 64  # Transformer patch size (samples)
    embed_dim: int = 768  # Embedding dimension
    num_heads: int = 12  # Attention heads
    mlp_ratio: float = 4.0  # MLP hidden dim ratio
    drop_rate: float = 0.0  # Dropout rate
    attn_drop_rate: float = 0.0  # Attention dropout
    
    # Training Parameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    max_epochs: int = 100
    batch_size: int = 32
    
    # Data Augmentation
    time_mask_ratio: float = 0.1  # Temporal masking
    channel_mask_ratio: float = 0.1  # Channel dropout
    noise_level: float = 0.01  # Gaussian noise std
    
    # Loss Weights
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    contrastive_weight: float = 0.1


class EEGPTOptimizer:
    """Bayesian optimization for EEGPT hyperparameters."""
    
    def __init__(
        self,
        train_data: TUABDataset,
        val_data: TUABDataset,
        n_trials: int = 100,
        n_jobs: int = 4
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters
        hparams = EEGPTHyperparameters(
            patch_size=trial.suggest_categorical("patch_size", [32, 64, 128]),
            embed_dim=trial.suggest_categorical("embed_dim", [512, 768, 1024]),
            num_heads=trial.suggest_categorical("num_heads", [8, 12, 16]),
            mlp_ratio=trial.suggest_float("mlp_ratio", 2.0, 8.0),
            drop_rate=trial.suggest_float("drop_rate", 0.0, 0.3),
            attn_drop_rate=trial.suggest_float("attn_drop_rate", 0.0, 0.3),
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            weight_decay=trial.suggest_loguniform("weight_decay", 1e-6, 1e-1),
            warmup_epochs=trial.suggest_int("warmup_epochs", 0, 20),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            time_mask_ratio=trial.suggest_float("time_mask_ratio", 0.0, 0.3),
            channel_mask_ratio=trial.suggest_float("channel_mask_ratio", 0.0, 0.3),
            noise_level=trial.suggest_float("noise_level", 0.0, 0.05),
            reconstruction_weight=trial.suggest_float("reconstruction_weight", 0.1, 2.0),
            classification_weight=trial.suggest_float("classification_weight", 0.5, 2.0),
            contrastive_weight=trial.suggest_float("contrastive_weight", 0.0, 0.5)
        )
        
        # Create model with suggested hyperparameters
        model = create_eegpt_model(hparams)
        
        # Create trainer with pruning callback
        trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_auroc"),
                EarlyStopping(monitor="val_auroc", mode="max", patience=10)
            ],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision=16,  # Mixed precision for speed
            gradient_clip_val=1.0,
            accumulate_grad_batches=4 if hparams.batch_size < 32 else 1
        )
        
        # Train model
        trainer.fit(
            model,
            train_dataloaders=self.train_data,
            val_dataloaders=self.val_data
        )
        
        # Return validation AUROC (to maximize)
        return trainer.callback_metrics["val_auroc"].item()
    
    def optimize(self) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        # Create study with storage for parallelization
        study = optuna.create_study(
            study_name="eegpt_optimization",
            direction="maximize",
            storage="sqlite:///eegpt_optimization.db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20
            )
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            gc_after_trial=True
        )
        
        # Log results
        print(f"Best AUROC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        # Save detailed results
        self._save_optimization_results(study)
        
        return study.best_params
```

### 2. AutoReject Parameter Tuning

#### AutoReject Hyperparameters
```python
# src/brain_go_brrr/optimization/autoreject_tuning.py

@dataclass
class AutoRejectHyperparameters:
    """AutoReject hyperparameter configuration."""
    
    # Thresholds
    n_interpolate: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    consensus: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Algorithm parameters
    cv: int = 5  # Cross-validation folds
    n_jobs: int = -1  # Parallel jobs
    random_state: int = 42
    
    # Channel rejection
    channel_threshold: float = 0.4  # Max bad channel ratio
    min_channels: int = 15  # Minimum required channels
    
    # Epoch rejection
    reject_criteria: Dict[str, float] = field(default_factory=lambda: {
        "eeg": 200e-6,  # 200 µV
        "eog": 250e-6   # 250 µV
    })


class AutoRejectOptimizer:
    """Grid search optimization for AutoReject."""
    
    def __init__(self, test_data: List[mne.io.Raw]):
        self.test_data = test_data
        self.results = {}
        
    def optimize_thresholds(self) -> AutoRejectHyperparameters:
        """Optimize rejection thresholds."""
        best_agreement = 0
        best_params = None
        
        # Grid search over thresholds
        for n_interp in [1, 2, 3, 4, 5]:
            for consensus in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for eeg_thresh in [150e-6, 200e-6, 250e-6, 300e-6]:
                    params = AutoRejectHyperparameters(
                        n_interpolate=list(range(1, n_interp + 1)),
                        consensus=[consensus],
                        reject_criteria={"eeg": eeg_thresh}
                    )
                    
                    # Evaluate on test set
                    agreement = self._evaluate_params(params)
                    
                    if agreement > best_agreement:
                        best_agreement = agreement
                        best_params = params
                    
                    # Log results
                    self.results[(n_interp, consensus, eeg_thresh)] = agreement
        
        print(f"Best agreement: {best_agreement:.3f}")
        print(f"Best params: {best_params}")
        
        return best_params
    
    def _evaluate_params(self, params: AutoRejectHyperparameters) -> float:
        """Evaluate parameters against expert annotations."""
        agreements = []
        
        for raw in self.test_data:
            # Create epochs
            epochs = make_fixed_length_epochs(raw, duration=4.0)
            
            # Apply AutoReject
            ar = AutoReject(
                n_interpolate=params.n_interpolate,
                consensus=params.consensus,
                cv=params.cv,
                n_jobs=params.n_jobs,
                random_state=params.random_state
            )
            
            _, reject_log = ar.fit_transform(epochs, return_log=True)
            
            # Compare with expert annotations
            if hasattr(epochs, 'expert_bad_epochs'):
                agreement = np.mean(
                    reject_log.bad_epochs == epochs.expert_bad_epochs
                )
                agreements.append(agreement)
        
        return np.mean(agreements)
```

### 3. Linear Probe Optimization

#### Classification Head Tuning
```python
# src/brain_go_brrr/optimization/linear_probe_tuning.py

@dataclass
class LinearProbeHyperparameters:
    """Linear probe hyperparameter configuration."""
    
    # Architecture
    hidden_dim: Optional[int] = None  # None for single layer
    use_batch_norm: bool = True
    dropout_rate: float = 0.5
    activation: str = "relu"  # relu, gelu, swish
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine, step, exponential
    warmup_steps: int = 500
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_prob: float = 0.0
    
    # Class balancing
    use_class_weights: bool = True
    focal_loss_gamma: float = 0.0  # 0 = standard CE
    
    # Optimization
    optimizer: str = "adamw"  # adamw, sgd, lamb
    gradient_clip: float = 1.0


class LinearProbeOptimizer:
    """Optimize linear probe for abnormality detection."""
    
    def __init__(
        self,
        feature_extractor: EEGPTModel,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.feature_extractor = feature_extractor
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective for linear probe."""
        # Suggest architecture
        use_hidden = trial.suggest_categorical("use_hidden", [True, False])
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512) if use_hidden else None
        
        hparams = LinearProbeHyperparameters(
            hidden_dim=hidden_dim,
            use_batch_norm=trial.suggest_categorical("use_batch_norm", [True, False]),
            dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.7),
            activation=trial.suggest_categorical("activation", ["relu", "gelu", "swish"]),
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            weight_decay=trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
            scheduler=trial.suggest_categorical("scheduler", ["cosine", "step", "exponential"]),
            warmup_steps=trial.suggest_int("warmup_steps", 0, 1000),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.3),
            mixup_alpha=trial.suggest_float("mixup_alpha", 0.0, 0.4),
            use_class_weights=trial.suggest_categorical("use_class_weights", [True, False]),
            focal_loss_gamma=trial.suggest_float("focal_loss_gamma", 0.0, 2.0),
            optimizer=trial.suggest_categorical("optimizer", ["adamw", "sgd"])
        )
        
        # Create probe
        probe = create_linear_probe(
            input_dim=768,  # EEGPT feature dimension
            num_classes=2,
            hparams=hparams
        )
        
        # Train probe
        best_auroc = train_probe_with_early_stopping(
            probe=probe,
            feature_extractor=self.feature_extractor,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            hparams=hparams,
            trial=trial  # For pruning
        )
        
        return best_auroc
```

### 4. Sleep Staging Optimization

#### YASA Parameter Tuning
```python
# src/brain_go_brrr/optimization/sleep_tuning.py

@dataclass
class SleepStagingHyperparameters:
    """Sleep staging hyperparameter configuration."""
    
    # Feature extraction
    freq_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "sigma": (12.0, 16.0),
        "beta": (16.0, 30.0),
        "gamma": (30.0, 50.0)
    })
    
    # YASA parameters
    eeg_name: str = "C3"  # Primary EEG channel
    eog_name: Optional[str] = "EOG"
    emg_name: Optional[str] = "EMG"
    
    # Classifier parameters
    classifier: str = "lgb"  # lgb, rf, xgb
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    
    # Post-processing
    min_bout_duration: Dict[str, int] = field(default_factory=lambda: {
        "N1": 1,  # Minimum 1 epoch (30s)
        "N2": 2,
        "N3": 2,
        "REM": 3,
        "W": 1
    })
    
    # Smoothing
    use_hmm_smoothing: bool = True
    transition_matrix: Optional[np.ndarray] = None


class SleepOptimizer:
    """Optimize sleep staging parameters."""
    
    def optimize_freq_bands(self) -> Dict[str, Tuple[float, float]]:
        """Optimize frequency band definitions."""
        best_bands = {}
        
        # Optimize each band independently
        for band_name in ["delta", "theta", "alpha", "sigma", "beta"]:
            best_acc = 0
            best_range = None
            
            # Grid search over frequency ranges
            for low_freq in np.arange(0.5, 30, 0.5):
                for high_freq in np.arange(low_freq + 1, 50, 1):
                    # Test this frequency range
                    test_bands = self.default_bands.copy()
                    test_bands[band_name] = (low_freq, high_freq)
                    
                    # Evaluate on validation set
                    acc = self._evaluate_bands(test_bands)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_range = (low_freq, high_freq)
            
            best_bands[band_name] = best_range
            print(f"{band_name}: {best_range} (acc: {best_acc:.3f})")
        
        return best_bands
```

## Multi-Objective Optimization

### Pareto-Optimal Solutions
```python
# src/brain_go_brrr/optimization/multiobjective.py
import optuna
from optuna.multi_objective import create_study

class MultiObjectiveOptimizer:
    """Multi-objective optimization for accuracy vs efficiency."""
    
    def __init__(self, dataset: TUABDataset):
        self.dataset = dataset
        
    def objective(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """Multi-objective: maximize accuracy, minimize latency & memory."""
        # Model hyperparameters affecting size/speed
        model_size = trial.suggest_categorical("model_size", ["small", "base", "large"])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        use_mixed_precision = trial.suggest_categorical("mixed_precision", [True, False])
        
        # Create model variant
        if model_size == "small":
            model = create_small_eegpt(embed_dim=384, num_heads=6)
        elif model_size == "base":
            model = create_base_eegpt(embed_dim=768, num_heads=12)
        else:
            model = create_large_eegpt(embed_dim=1024, num_heads=16)
        
        # Evaluate objectives
        accuracy = self._evaluate_accuracy(model, batch_size)
        latency = self._measure_latency(model, batch_size, use_mixed_precision)
        memory = self._measure_memory(model, batch_size)
        
        # Return objectives (accuracy to maximize, others to minimize)
        return -accuracy, latency, memory  # Optuna minimizes by default
    
    def find_pareto_optimal(self, n_trials: int = 200) -> List[Dict]:
        """Find Pareto-optimal configurations."""
        study = create_study(
            directions=["minimize", "minimize", "minimize"],
            study_name="pareto_optimization"
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        # Extract Pareto front
        pareto_trials = []
        for trial in study.best_trials:
            config = {
                "params": trial.params,
                "accuracy": -trial.values[0],
                "latency_ms": trial.values[1],
                "memory_mb": trial.values[2]
            }
            pareto_trials.append(config)
        
        return pareto_trials
```

## Hyperparameter Schedules

### Learning Rate Scheduling
```python
# src/brain_go_brrr/optimization/schedulers.py

class CosineAnnealingWithWarmup:
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * scale
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


class OneCycleLR:
    """One-cycle learning rate schedule."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos"
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.current_step = 0
```

## Experiment Tracking

### Weights & Biases Integration
```python
# src/brain_go_brrr/optimization/tracking.py
import wandb
from typing import Dict, Any, Optional

class ExperimentTracker:
    """Track hyperparameter optimization experiments."""
    
    def __init__(
        self,
        project: str = "brain-go-brrr",
        entity: Optional[str] = None,
        tags: List[str] = None
    ):
        self.project = project
        self.entity = entity
        self.tags = tags or []
        
    def log_trial(
        self,
        trial_id: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Any]] = None
    ):
        """Log single optimization trial."""
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"trial_{trial_id}",
            tags=self.tags + ["hyperopt"],
            config=hyperparameters
        )
        
        # Log metrics
        wandb.log(metrics)
        
        # Log artifacts
        if artifacts:
            for name, artifact in artifacts.items():
                if isinstance(artifact, plt.Figure):
                    wandb.log({name: wandb.Image(artifact)})
                elif isinstance(artifact, pd.DataFrame):
                    wandb.log({name: wandb.Table(dataframe=artifact)})
        
        wandb.finish()
    
    def log_optimization_study(
        self,
        study: optuna.Study,
        name: str = "optimization_study"
    ):
        """Log complete Optuna study."""
        # Create summary
        summary = {
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number
        }
        
        # Create visualizations
        figs = {
            "optimization_history": optuna.visualization.plot_optimization_history(study),
            "param_importances": optuna.visualization.plot_param_importances(study),
            "parallel_coordinate": optuna.visualization.plot_parallel_coordinate(study),
            "slice": optuna.visualization.plot_slice(study)
        }
        
        # Log to wandb
        run = wandb.init(
            project=self.project,
            name=name,
            tags=self.tags + ["study_summary"]
        )
        
        wandb.log(summary)
        for fig_name, fig in figs.items():
            wandb.log({fig_name: wandb.Image(fig)})
        
        # Save study database
        study_artifact = wandb.Artifact(
            f"{name}_study",
            type="optuna_study",
            description=f"Optuna study with {len(study.trials)} trials"
        )
        study_artifact.add_file(f"{name}.db")
        run.log_artifact(study_artifact)
        
        wandb.finish()
```

## Best Practices for Hyperparameter Optimization

### 1. Systematic Search Strategy
```python
def hyperparameter_search_pipeline():
    """Complete hyperparameter optimization pipeline."""
    
    # Step 1: Quick random search for rough range
    rough_params = random_search(
        n_trials=50,
        param_ranges={
            "learning_rate": (1e-5, 1e-1),
            "batch_size": [16, 32, 64, 128],
            "dropout": (0.0, 0.5)
        }
    )
    
    # Step 2: Refined Bayesian optimization
    refined_params = bayesian_optimization(
        n_trials=200,
        initial_params=rough_params,
        param_ranges={
            "learning_rate": (rough_params["learning_rate"] * 0.1, rough_params["learning_rate"] * 10),
            "batch_size": rough_params["batch_size"],
            "dropout": (max(0, rough_params["dropout"] - 0.1), min(0.5, rough_params["dropout"] + 0.1))
        }
    )
    
    # Step 3: Final grid search for fine-tuning
    final_params = grid_search(
        param_grid={
            "learning_rate": [refined_params["learning_rate"] * x for x in [0.5, 1.0, 2.0]],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "warmup_ratio": [0.0, 0.05, 0.1]
        }
    )
    
    return final_params
```

### 2. Cross-Validation Strategy
```python
class RobustCrossValidation:
    """Robust cross-validation for hyperparameter selection."""
    
    def __init__(self, n_folds: int = 5, stratified: bool = True):
        self.n_folds = n_folds
        self.stratified = stratified
        
    def evaluate(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate parameters with cross-validation."""
        if self.stratified:
            kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        metrics = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_fn(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict_proba(X_val)[:, 1]
            fold_metrics = {
                "auroc": roc_auc_score(y_val, y_pred),
                "auprc": average_precision_score(y_val, y_pred),
                "accuracy": accuracy_score(y_val, y_pred > 0.5)
            }
            metrics.append(fold_metrics)
        
        # Aggregate results
        return {
            metric: {
                "mean": np.mean([m[metric] for m in metrics]),
                "std": np.std([m[metric] for m in metrics]),
                "min": np.min([m[metric] for m in metrics]),
                "max": np.max([m[metric] for m in metrics])
            }
            for metric in metrics[0].keys()
        }
```

### 3. Early Stopping and Pruning
```python
class AdaptivePruner:
    """Adaptive pruning for efficient hyperparameter search."""
    
    def __init__(
        self,
        min_epochs: int = 10,
        patience: int = 5,
        percentile: float = 50.0
    ):
        self.min_epochs = min_epochs
        self.patience = patience
        self.percentile = percentile
        self.trial_history = []
        
    def should_prune(
        self,
        trial_id: int,
        epoch: int,
        current_score: float
    ) -> bool:
        """Decide whether to prune trial."""
        if epoch < self.min_epochs:
            return False
        
        # Get scores at same epoch from completed trials
        scores_at_epoch = [
            h["scores"][epoch] 
            for h in self.trial_history 
            if len(h["scores"]) > epoch
        ]
        
        if len(scores_at_epoch) < 5:  # Need enough history
            return False
        
        # Prune if below percentile
        threshold = np.percentile(scores_at_epoch, self.percentile)
        return current_score < threshold
```

## Hyperparameter Configuration Files

### YAML Configuration
```yaml
# configs/hyperparameters/eegpt_best.yaml
model:
  architecture:
    patch_size: 64
    embed_dim: 768
    num_heads: 12
    mlp_ratio: 4.0
    drop_rate: 0.1
    attn_drop_rate: 0.1
    
training:
  optimizer:
    type: "AdamW"
    learning_rate: 0.001
    weight_decay: 0.05
    betas: [0.9, 0.999]
    
  scheduler:
    type: "CosineAnnealingWithWarmup"
    warmup_epochs: 10
    min_lr: 1e-6
    
  regularization:
    label_smoothing: 0.1
    mixup_alpha: 0.2
    cutmix_prob: 0.0
    gradient_clip: 1.0
    
  data_augmentation:
    time_shift: 0.1
    amplitude_scale: 0.1
    gaussian_noise: 0.01
    channel_dropout: 0.1

evaluation:
  metrics: ["auroc", "accuracy", "sensitivity", "specificity"]
  save_best_metric: "auroc"
  early_stopping_patience: 20
```

## Automated Hyperparameter Selection

### AutoML Integration
```python
# src/brain_go_brrr/optimization/automl.py
from autosklearn.classification import AutoSklearnClassifier

class AutoMLOptimizer:
    """Automated machine learning for hyperparameter selection."""
    
    def __init__(
        self,
        time_left_for_this_task: int = 3600,  # 1 hour
        per_run_time_limit: int = 300,  # 5 minutes
        n_jobs: int = 4
    ):
        self.time_limit = time_left_for_this_task
        self.run_limit = per_run_time_limit
        self.n_jobs = n_jobs
        
    def optimize_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Use AutoML to find best classifier configuration."""
        automl = AutoSklearnClassifier(
            time_left_for_this_task=self.time_limit,
            per_run_time_limit=self.run_limit,
            n_jobs=self.n_jobs,
            metric=autosklearn.metrics.roc_auc,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
        )
        
        # Fit AutoML
        automl.fit(X_train, y_train)
        
        # Get best model configuration
        best_model = automl.get_models_with_weights()[0][1]
        best_params = best_model.get_params()
        
        # Evaluate on validation set
        y_pred = automl.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, y_pred)
        
        return {
            "best_params": best_params,
            "val_auroc": val_auroc,
            "ensemble_weights": automl.get_models_with_weights()
        }
```

## Hyperparameter Optimization Schedule

### Progressive Optimization
```python
def progressive_hyperparameter_optimization():
    """Progressive optimization from coarse to fine."""
    
    # Phase 1: Architecture search (1 day)
    architecture_params = optimize_architecture(
        search_space={
            "model_size": ["small", "base", "large"],
            "patch_size": [32, 64, 128],
            "use_pretrained": [True, False]
        },
        n_trials=50,
        time_budget=86400  # 24 hours
    )
    
    # Phase 2: Training hyperparameters (2 days)
    training_params = optimize_training(
        fixed_architecture=architecture_params,
        search_space={
            "learning_rate": (1e-5, 1e-2),
            "batch_size": [16, 32, 64],
            "optimizer": ["adamw", "sgd", "lamb"],
            "scheduler": ["cosine", "onecycle", "exponential"]
        },
        n_trials=200,
        time_budget=172800  # 48 hours
    )
    
    # Phase 3: Regularization fine-tuning (1 day)
    regularization_params = optimize_regularization(
        fixed_architecture=architecture_params,
        fixed_training=training_params,
        search_space={
            "dropout": (0.0, 0.5),
            "weight_decay": (1e-6, 1e-2),
            "label_smoothing": (0.0, 0.3),
            "mixup_alpha": (0.0, 0.4)
        },
        n_trials=100,
        time_budget=86400  # 24 hours
    )
    
    return {
        "architecture": architecture_params,
        "training": training_params,
        "regularization": regularization_params
    }
```

## Conclusion

This comprehensive hyperparameter optimization guide ensures that Brain-Go-Brrr achieves optimal performance across all components. By systematically exploring the hyperparameter space using Bayesian optimization, multi-objective optimization, and automated techniques, we can find configurations that balance accuracy, speed, and resource usage for production deployment.