"""
Progressive training strategy hooks for heatmap-based pose estimation.

These hooks implement sophisticated training strategies to improve convergence:
1. LossWeightWarmupHook: Delayed loss-weight warmup for arbitrary losses
2. LayerDecayLRHook: Layer-wise learning rate scheduling
"""

import math
from typing import Optional, Sequence
from mmengine.hooks import Hook
from mmwave3d.registry import HOOKS


@HOOKS.register_module()
class LossWeightWarmupHook(Hook):
    """
    Warm up an arbitrary loss weight for a fixed number of iterations.

    This hook targets a loss attribute by name (usually on model.head), then
    gradually increases its weight from init_weight to final_weight using
    specified warmup strategy.

    Args:
        loss_name (str): Attribute name of the loss to warm up. If found on
            model.head, it will be used; otherwise model is searched.
        warmup_iters (int): Number of iterations to use for warmup.
        init_weight (float): Initial weight at start of warmup. Default: 0.0
        final_weight (float): Final weight after warmup. Default: 1.0
        warmup_type (str): Type of warmup strategy. Options:
            - 'linear': Linear increase from init_weight to final_weight
            - 'cosine': Cosine annealing from init_weight to final_weight (smoother)
            Default: 'cosine'
        loss_warmups (list[dict], optional): Per-loss warmup configs. Each dict
            supports keys: loss_name, warmup_iters, init_weight, final_weight, warmup_type.
            If provided, overrides the single-loss arguments.

    Example:
        >>> # Single loss with cosine warmup
        >>> custom_hooks = [
        ...     dict(type='LossWeightWarmupHook',
        ...          loss_name='coord_loss_weight',
        ...          warmup_iters=3000,
        ...          init_weight=0.0,
        ...          final_weight=5.0,
        ...          warmup_type='cosine')
        ... ]
        >>> # Multiple losses with different strategies
        >>> custom_hooks = [
        ...     dict(type='LossWeightWarmupHook',
        ...          loss_warmups=[
        ...              dict(loss_name='coord_loss_weight',
        ...                   warmup_iters=3000,
        ...                   init_weight=0.0,
        ...                   final_weight=5.0,
        ...                   warmup_type='cosine'),
        ...              dict(loss_name='loss_heatmap',
        ...                   warmup_iters=1000,
        ...                   init_weight=1.0,
        ...                   final_weight=10.0,
        ...                   warmup_type='linear'),
        ...          ])
        ... ]
    """

    priority = 'LOW'

    def __init__(self,
                 loss_name: Optional[str] = None,
                 warmup_iters: int = 3000,
                 init_weight: float = 0.0,
                 final_weight: float = 1.0,
                 warmup_type: str = 'cosine',
                 loss_warmups: Optional[list] = None):
        super().__init__()
        if loss_warmups:
            self._entries = []
            for entry in loss_warmups:
                if 'loss_name' not in entry:
                    raise ValueError("Each loss_warmups entry must include loss_name.")
                self._entries.append({
                    'loss_name': entry['loss_name'],
                    'warmup_iters': entry.get('warmup_iters', warmup_iters),
                    'init_weight': entry.get('init_weight', init_weight),
                    'final_weight': entry.get('final_weight', final_weight),
                    'warmup_type': entry.get('warmup_type', warmup_type),
                    'logged': False
                })
        else:
            if not loss_name:
                raise ValueError("loss_name is required when loss_warmups is not provided.")
            self._entries = [{
                'loss_name': loss_name,
                'warmup_iters': warmup_iters,
                'init_weight': init_weight,
                'final_weight': final_weight,
                'warmup_type': warmup_type,
                'logged': False
            }]

    def _resolve_loss_target(self, model, loss_name: str):
        candidates = [loss_name]
        if not loss_name.endswith('_fn'):
            candidates.append(f"{loss_name}_fn")
        if loss_name.startswith('loss_'):
            candidates.append(f"{loss_name}_fn")
        for name in candidates:
            if hasattr(model, 'head') and hasattr(model.head, name):
                return model.head, name
            if hasattr(model, name):
                return model, name
        raise AttributeError(
            f"Loss attribute '{loss_name}' not found on model.head or model."
        )

    def _set_loss_weight(self, container, attr_name: str, weight: float):
        target = getattr(container, attr_name)
        if hasattr(target, 'loss_weight'):
            target.loss_weight = weight
            return
        if isinstance(target, (int, float)):
            setattr(container, attr_name, weight)
            return
        raise TypeError(
            f"Loss attribute '{attr_name}' does not support weight warmup. "
            "Expected a numeric field or an object with loss_weight."
        )

    def _compute_warmup_weight(self, cur_iter: int, entry: dict) -> float:
        """Compute the current weight based on warmup strategy."""
        if cur_iter >= entry['warmup_iters']:
            return entry['final_weight']
        
        init_weight = entry['init_weight']
        final_weight = entry['final_weight']
        warmup_iters = entry['warmup_iters']
        warmup_type = entry['warmup_type']
        
        # Progress ratio: 0.0 at start, 1.0 at end
        progress = cur_iter / warmup_iters
        
        if warmup_type == 'linear':
            # Linear interpolation
            weight = init_weight + (final_weight - init_weight) * progress
        elif warmup_type == 'cosine':
            # Cosine annealing: starts slow, accelerates in middle, slows down at end
            # Using cosine formula: weight = init + (final - init) * (1 - cos(pi * progress)) / 2
            weight = init_weight + (final_weight - init_weight) * (1 - math.cos(math.pi * progress)) / 2
        else:
            raise ValueError(f"Unknown warmup_type: {warmup_type}. Supported: 'linear', 'cosine'")
        
        return weight

    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        """Adjust target loss weight before each training iteration."""
        cur_iter = runner.iter

        # Handle DDP wrapped model
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        for entry in self._entries:
            container, attr_name = self._resolve_loss_target(model, entry['loss_name'])
            
            # Compute current weight based on warmup strategy
            weight = self._compute_warmup_weight(cur_iter, entry)
            self._set_loss_weight(container, attr_name, weight)
            
            # Log when warmup finishes
            if cur_iter >= entry['warmup_iters'] and not entry['logged']:
                runner.logger.info(
                    f"[LossWeightWarmupHook] Iter {cur_iter}: Warmup finished for "
                    f"'{entry['loss_name']}' (weight: {entry['init_weight']} -> "
                    f"{entry['final_weight']}, strategy: {entry['warmup_type']})"
                )
                entry['logged'] = True


@HOOKS.register_module()
class CoordLossWarmupHook(LossWeightWarmupHook):
    """Backward-compatible wrapper for coordinate loss warmup."""

    def __init__(self,
                 warmup_iters: int = 3000,
                 init_weight: float = 0.0,
                 final_weight: float = 5.0):
        super().__init__(
            loss_name='coord_loss_weight',
            warmup_iters=warmup_iters,
            init_weight=init_weight,
            final_weight=final_weight
        )


@HOOKS.register_module()
class LayerDecayLRHook(Hook):
    """
    Apply layer-wise learning rate decay for different model components.
    
    Training Strategy:
    -----------------
    Different network modules have different learning dynamics:
    
    1. Backbone (Voxelizer/Data Preprocessor):
       - Handles low-level feature extraction from point clouds
       - Should be trained conservatively to avoid overfitting
       - Gets lowest LR: base_lr * backbone_lr_mult
       
    2. Transformer (Feature Encoder):
       - Learns spatial-temporal relationships
       - Needs moderate learning rate for stable attention learning
       - Gets medium LR: base_lr * transformer_lr_mult
       
    3. Head (Task-Specific Layers):
       - Directly predicts heatmaps and coordinates
       - Needs aggressive learning for fast task adaptation
       - Gets full LR: base_lr * 1.0
    
    Progressive Decay (Optional):
    ---------------------------
    After decay_start_iter, gradually reduces backbone LR further to prevent
    overfitting in later training stages:
    
    - Before decay_start_iter: LR = base_lr * backbone_lr_mult
    - After decay_start_iter: LR linearly decays to base_lr * backbone_lr_mult * decay_factor
    - At max_iters: LR = base_lr * backbone_lr_mult * decay_factor
    
    Rationale:
    ---------
    1. Pre-trained or frozen-like backbone: Lower LR preserves learned features
    2. Task-specific head: Higher LR enables fast adaptation to new task
    3. Progressive decay: Prevents late-stage overfitting while maintaining performance
    
    Args:
        backbone_lr_mult (float): LR multiplier for backbone. Default: 0.1
        transformer_lr_mult (float): LR multiplier for transformer. Default: 0.5
        decay_start_iter (int, optional): When to start progressive decay.
            If None, no progressive decay is applied. Default: None
        decay_factor (float): Final LR multiplier for backbone at end of training.
            Only effective when decay_start_iter is set. Default: 0.1
    
    Example:
        >>> custom_hooks = [
        ...     dict(type='LayerDecayLRHook',
        ...          backbone_lr_mult=0.1,      # Backbone: 10% LR
        ...          transformer_lr_mult=0.5,   # Transformer: 50% LR
        ...          decay_start_iter=15000,    # Start decay at iter 15k
        ...          decay_factor=0.1)          # Final: 1% LR for backbone
        ... ]
    """
    
    priority = 'NORMAL'
    
    def __init__(self,
                 backbone_lr_mult: float = 0.1,
                 transformer_lr_mult: float = 0.5,
                 decay_start_iter: Optional[int] = None,
                 decay_factor: float = 0.1):
        super().__init__()
        self.backbone_lr_mult = backbone_lr_mult
        self.transformer_lr_mult = transformer_lr_mult
        self.decay_start_iter = decay_start_iter
        self.decay_factor = decay_factor
        self._param_groups_configured = False
        self._initial_backbone_lr = None

    def _get_head_lr(self, optimizer):
        """Fetch current head LR (after schedulers) if available."""
        for group in optimizer.param_groups:
            if group.get('name') == 'head':
                return group.get('lr')
        if optimizer.param_groups:
            return optimizer.param_groups[-1].get('lr')
        return optimizer.defaults.get('lr', 1e-4)

    def before_train(self, runner):
        """Setup parameter groups with different LR multipliers."""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        optimizer = runner.optim_wrapper.optimizer
        
        # Get base LR from optimizer
        if len(optimizer.param_groups) > 0:
            base_lr = optimizer.param_groups[0]['lr']
        else:
            base_lr = optimizer.defaults.get('lr', 1e-4)
        
        # Clear existing param groups and rebuild
        optimizer.param_groups.clear()
        
        # Group 1: Backbone (voxelizer/data_preprocessor) - lowest LR
        if hasattr(model, 'voxelizer'):
            backbone_params = list(model.voxelizer.parameters())
            if backbone_params:
                backbone_lr = base_lr * self.backbone_lr_mult
                self._initial_backbone_lr = backbone_lr
                optimizer.add_param_group({
                    'params': backbone_params,
                    'lr': backbone_lr,
                    'name': 'backbone'
                })
                runner.logger.info(
                    f"[LayerDecayLRHook] Backbone LR: {backbone_lr:.2e} "
                    f"(base_lr × {self.backbone_lr_mult})"
                )
            else:
                runner.logger.info(
                    "[LayerDecayLRHook] Voxelizer has no trainable parameters; "
                    "skipping backbone LR group."
                )
        
        # Group 2: Transformer - medium LR
        if hasattr(model, 'transformer'):
            transformer_lr = base_lr * self.transformer_lr_mult
            optimizer.add_param_group({
                'params': model.transformer.parameters(),
                'lr': transformer_lr,
                'name': 'transformer'
            })
            runner.logger.info(
                f"[LayerDecayLRHook] Transformer LR: {transformer_lr:.2e} "
                f"(base_lr × {self.transformer_lr_mult})"
            )
        
        # Group 3: Head - full LR
        if hasattr(model, 'head'):
            optimizer.add_param_group({
                'params': model.head.parameters(),
                'lr': base_lr,
                'name': 'head'
            })
            runner.logger.info(
                f"[LayerDecayLRHook] Head LR: {base_lr:.2e} (base_lr × 1.0)"
            )

        # Log progressive decay settings
        if self.decay_start_iter is not None:
            runner.logger.info(
                f"[LayerDecayLRHook] Progressive decay enabled: "
                f"starts at iter {self.decay_start_iter}, "
                f"final backbone LR multiplier: "
                f"{self.backbone_lr_mult * self.decay_factor:.4f}"
            )

        self._param_groups_configured = True

    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        """Apply progressive decay to backbone if enabled."""
        if self.decay_start_iter is None or not self._param_groups_configured:
            return

        cur_iter = runner.iter
        if cur_iter < self.decay_start_iter:
            return

        # Calculate decay progress (0.0 at decay_start_iter, 1.0 at max_iters)
        max_iters = runner.max_iters
        progress = (cur_iter - self.decay_start_iter) / max(
            max_iters - self.decay_start_iter, 1
        )
        progress = min(progress, 1.0)

        # Linear decay from initial multiplier to (initial * decay_factor)
        current_mult = self.backbone_lr_mult * (
            1.0 - progress * (1.0 - self.decay_factor)
        )

        optimizer = runner.optim_wrapper.optimizer

        # Update backbone param group (assumed to be index 0)
        if len(optimizer.param_groups) > 0:
            # Use current (scheduled) head LR as base
            base_lr = self._get_head_lr(optimizer)
            new_backbone_lr = base_lr * current_mult

            # Update backbone LR
            optimizer.param_groups[0]['lr'] = new_backbone_lr


@HOOKS.register_module()
class FreezeModelExceptHook(Hook):
    """Freeze all model parameters except modules under selected prefixes."""

    priority = 'VERY_HIGH'

    def __init__(self,
                 trainable_prefixes: Sequence[str],
                 verbose: bool = True):
        super().__init__()
        if not trainable_prefixes:
            raise ValueError('trainable_prefixes must contain at least one prefix.')
        self.trainable_prefixes = tuple(trainable_prefixes)
        self.verbose = bool(verbose)

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            keep_trainable = any(
                name.startswith(prefix) for prefix in self.trainable_prefixes
            )
            param.requires_grad = keep_trainable
            numel = param.numel()
            total_params += numel
            if keep_trainable:
                trainable_params += numel

        if self.verbose:
            frozen_params = total_params - trainable_params
            pct = (trainable_params / max(total_params, 1)) * 100.0
            runner.logger.info(
                '[FreezeModelExceptHook] trainable_prefixes=%s, '
                'trainable=%d, frozen=%d (%.2f%% trainable)',
                self.trainable_prefixes, trainable_params, frozen_params, pct
            )
