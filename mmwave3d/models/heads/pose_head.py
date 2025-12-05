from typing import Dict, List, Optional, Tuple
import time

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS
from mmengine.logging import MMLogger

@MODELS.register_module()
class MmWavePoseHead(nn.Module):
    """MLP head predicting 4-person 3D keypoints (17 per person) & scores."""

    def __init__(self,
                 in_dim: int,
                 num_person: int = 4,
                 num_joints: int = 17,
                 mlp_hidden_dim: int = 512,
                 mlp_depth: int = 2,
                 loss_mpjpe: Optional[Dict] = dict(type='MPJPELoss', loss_weight=1.0),
                 loss_human_count: Optional[Dict] = dict(
                     type='HumanCountLoss', threshold=0.5, min_joints=9,
                     loss_weight=1.0)) -> None:
        super().__init__()
        self.num_person = num_person
        self.num_joints = num_joints
        self.loss_mpjpe = MODELS.build(loss_mpjpe) if loss_mpjpe else None
        self.loss_human_count = (
            MODELS.build(loss_human_count) if loss_human_count else None)

        mlp_layers = []
        dim = in_dim
        for _ in range(mlp_depth):
            mlp_layers += [nn.Linear(dim, mlp_hidden_dim), nn.ReLU(inplace=True)]
            dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.keypoint_head = nn.Linear(dim, num_person * num_joints * 3)
        self.score_head = nn.Linear(dim, num_person * num_joints)

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        # torch.cuda.synchronize()
        # start_time = time.time()
        feats = inputs['features']  # (B, N, C) from transformer
        pooled = feats.mean(dim=1)
        feat_emb = self.mlp(pooled)
        keypoints = self.keypoint_head(feat_emb).view(
            feats.size(0), self.num_person, self.num_joints, 3)
        scores = torch.sigmoid(self.score_head(feat_emb).view(
            feats.size(0), self.num_person, self.num_joints))
        outputs = dict(keypoints=keypoints, scores=scores)
        # torch.cuda.synchronize()
        # head_time = time.time()-start_time
        # logger = MMLogger.get_current_instance()
        # if logger is not None:
        #             logger.info(
        #                 'head timings - head: %.4fs',
        #                 head_time)

        if data_samples is not None:
            # torch.cuda.synchronize()
            # start_time = time.time()
            gt_kpts, mask = self._stack_gt(
                data_samples, keypoints.device, keypoints.size(0))
            # torch.cuda.synchronize()
            # if logger is not None:
            #         logger.info(
            #             'head gt timings - gt: %.4fs',
            #             time.time()-start_time)
            # torch.cuda.synchronize()
            # start_time= time.time()
            if gt_kpts is not None:
                loss_dict = self.loss(keypoints, scores, gt_kpts, mask)
                outputs.update(loss_dict)
                # torch.cuda.synchronize()
                # if logger is not None:
                #     logger.info(
                #         'head loss timings - loss: %.4fs',
                #         time.time()-start_time)
        return outputs

    def _stack_gt(self, data_samples: List, device: torch.device,
                  batch_size: int
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not data_samples or batch_size == 0:
            return None, None
        bsz = batch_size
        gt_batch = torch.zeros(
            (bsz, self.num_person, self.num_joints, 3), device=device)
        mask_batch = torch.zeros(
            (bsz, self.num_person, self.num_joints), dtype=torch.bool, device=device)
        processed = 0
        for idx, sample in enumerate(data_samples):
            if idx >= batch_size:
                break
            if not hasattr(sample, 'gt_keypoints_3d'):
                continue
            gt = sample.gt_keypoints_3d
            if gt is None:
                continue
            gt = gt.to(device)
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)
            if gt.shape[-1] > 3:
                coords = gt[..., :3]
                extra = gt[..., 3:]
            else:
                coords = gt
                extra = None
            persons = min(coords.shape[0], self.num_person)
            if persons == 0:
                continue
            gt_batch[idx, :persons] = coords[:persons]
            mask = coords[:persons].abs().sum(dim=-1) > 0
            if extra is not None:
                if extra.shape[-1] == 1:
                    extra_mask = extra[:persons, :, 0] > 0
                else:
                    extra_mask = extra[:persons].abs().sum(dim=-1) > 0
                mask = mask | extra_mask
            mask_batch[idx, :persons] = mask
            processed += 1
        if mask_batch.sum() == 0 or processed == 0:
            return None, None
        return gt_batch, mask_batch

    def loss(self, pred_kpts: torch.Tensor, pred_scores: torch.Tensor,
             gt_kpts: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = dict()
        # torch.cuda.synchronize()
        # start_time = time.time()
        if self.loss_mpjpe is not None:
            losses['loss_mpjpe'] = self.loss_mpjpe(pred_kpts, gt_kpts, mask)
            # torch.cuda.synchronize()
            # mpjpe_t = time.time()
            # mpjpe_time = mpjpe_t-start_time
        if self.loss_human_count is not None:
            losses['loss_human_count'] = self.loss_human_count(
                pred_scores, mask.float())
        #     torch.cuda.synchronize()
        #     human_t = time.time()
        #     human_time = human_t-mpjpe_t
        # logger = MMLogger.get_current_instance()
        # if logger is not None:
        #             logger.info(
        #                 'loss timings - mpjpe: %.4fs.  human: %.4fs',
        #                 mpjpe_time, human_time)
        return losses
