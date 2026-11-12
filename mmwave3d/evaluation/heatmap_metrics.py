import json
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from mmwave3d.registry import METRICS
from mmwave3d.utils import spherical_to_cartesian


@METRICS.register_module()
class HeatmapMPJPE(BaseMetric):
    default_prefix: Optional[str] = "heatmap"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        decode_from_heatmap: bool = True,
        softargmax_temperature: float = 3.0,
        point_cloud_range: Sequence[float] = [0.12, -0.524, -0.524, 6.0, 2.618, 0.524],
        analysis_out_dir: Optional[str] = None,
        low_mpjpe_thr: float = 0.06,
        high_mpjpe_thr: float = 0.10,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.decode_from_heatmap = decode_from_heatmap
        self.softargmax_temperature = softargmax_temperature
        self.analysis_out_dir = (
            Path(analysis_out_dir) if analysis_out_dir is not None else None
        )
        self.low_mpjpe_thr = float(low_mpjpe_thr)
        self.high_mpjpe_thr = float(high_mpjpe_thr)

        self.r_min, self.az_min, self.el_min = point_cloud_range[:3]
        self.r_max, self.az_max, self.el_max = point_cloud_range[3:]

    def _get_field(self, sample, name):
        if isinstance(sample, dict):
            return sample.get(name, None)
        if hasattr(sample, name):
            return getattr(sample, name)
        if hasattr(sample, "get"):
            try:
                return sample.get(name, None)
            except Exception:
                return None
        return None

    def _get_pred_keypoints(
        self, data_sample, sample_idx: int, logger: MMLogger
    ) -> Optional[torch.Tensor]:

        if self.decode_from_heatmap:
            heatmaps = self._get_field(data_sample, "pred_heatmaps")
            elevation_maps = self._get_field(data_sample, "pred_elevation_maps")
            re_heatmaps = self._get_field(data_sample, "pred_re_heatmaps")

            if heatmaps is not None and re_heatmaps is not None:
                return self.decode_keypoints_from_dual_heatmap(heatmaps, re_heatmaps)

            if heatmaps is not None and elevation_maps is not None:
                return self.decode_keypoints_from_heatmap(heatmaps, elevation_maps)

            logger.debug(
                f"Sample {sample_idx}: Missing heatmaps, using pre-decoded keypoints"
            )

        pred_keypoints = self._get_field(data_sample, "pred_keypoints_3d")
        if pred_keypoints is None:
            pred_keypoints = self._get_field(data_sample, "pred_keypoints")
        if pred_keypoints is None:
            pred_keypoints = self._get_field(data_sample, "keypoints")
        return pred_keypoints

    def _resolve_sample_name(self, data_sample, sample_idx: int) -> str:
        metainfo = getattr(data_sample, "metainfo", {})
        name = getattr(data_sample, "dump_name", None)
        if not name:
            try:
                name = data_sample["sample_name"]
            except Exception:
                name = None
        if not name and hasattr(metainfo, "get"):
            name = metainfo.get("sample_name")
        if name:
            return str(name)
        paths = (
            metainfo.get("radar_points_paths", None)
            if hasattr(metainfo, "get")
            else None
        )
        if paths:
            if isinstance(paths, list) and paths:
                center = len(paths) // 2
                path_str = paths[center]
            else:
                path_str = paths
            return Path(path_str).stem
        return f"sample_{sample_idx:06d}"

    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        logger = MMLogger.get_current_instance()

        for idx, data_sample in enumerate(data_samples):
            result = {}

            pred_keypoints = self._get_pred_keypoints(data_sample, idx, logger)
            sample_name = self._resolve_sample_name(data_sample, idx)

            gt_keypoints = self._get_field(data_sample, "gt_keypoints_3d")

            if pred_keypoints is None:
                logger.debug(
                    f"Sample {idx}: Missing pred_keypoints (checked: pred_keypoints_3d, pred_keypoints, keypoints)"
                )
                continue

            if gt_keypoints is None:
                logger.debug(f"Sample {idx}: Missing gt_keypoints_3d")
                continue

            pred_scores = self._get_field(data_sample, "pred_scores")
            if pred_scores is None:
                pred_scores = self._get_field(data_sample, "scores")

            result["pred_keypoints"] = pred_keypoints.cpu()
            result["gt_keypoints"] = gt_keypoints.cpu()
            result["sample_name"] = sample_name
            if pred_scores is not None:
                result["pred_scores"] = pred_scores.cpu()

            self.results.append(result)

    def decode_keypoints_from_dual_heatmap(
        self, heatmaps: torch.Tensor, re_heatmaps: torch.Tensor
    ) -> torch.Tensor:
        if heatmaps.dim() == 4:
            heatmaps = heatmaps.unsqueeze(0)
            re_heatmaps = re_heatmaps.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, P, J, H, W = heatmaps.shape
        E = re_heatmaps.size(-1)
        device = heatmaps.device

        if self.softargmax_temperature != 1.0:
            heatmaps = torch.pow(heatmaps.clamp(min=1e-8), self.softargmax_temperature)
            re_heatmaps = torch.pow(
                re_heatmaps.clamp(min=1e-8), self.softargmax_temperature
            )

        r_coords = torch.linspace(self.r_min, self.r_max, H, device=device)
        az_coords = torch.linspace(self.az_min, self.az_max, W, device=device)
        el_coords = torch.linspace(self.el_min, self.el_max, E, device=device)

        ra_range = heatmaps.sum(dim=-1)
        ra_az = heatmaps.sum(dim=-2)
        re_range = re_heatmaps.sum(dim=-1)
        re_el = re_heatmaps.sum(dim=-2)

        def _soft(dist, coords):
            total = dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return (dist * coords).sum(dim=-1, keepdim=True) / total

        r_pred = 0.5 * (_soft(ra_range, r_coords) + _soft(re_range, r_coords))
        el_pred = _soft(re_el, el_coords)

        sin_w = (torch.sin(az_coords) * ra_az).sum(dim=-1)
        cos_w = (torch.cos(az_coords) * ra_az).sum(dim=-1)
        az_pred = torch.atan2(sin_w, cos_w).unsqueeze(-1)

        keypoints = torch.cat([r_pred, az_pred, el_pred], dim=-1)
        return keypoints.squeeze(0) if squeeze_batch else keypoints

    def decode_keypoints_from_heatmap(
        self, heatmaps: torch.Tensor, elevation_maps: torch.Tensor
    ) -> torch.Tensor:

        if heatmaps.dim() == 4:
            heatmaps = heatmaps.unsqueeze(0)
            elevation_maps = elevation_maps.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, P, J, H, W = heatmaps.shape
        device = heatmaps.device

        if self.softargmax_temperature != 1.0:
            heatmaps = torch.pow(heatmaps, self.softargmax_temperature)

        heatmaps_flat = heatmaps.view(B, P, J, H * W)
        elevation_flat = elevation_maps.view(B, P, J, H * W)

        r_coords = torch.linspace(self.r_min, self.r_max, H, device=device)
        az_coords = torch.linspace(self.az_min, self.az_max, W, device=device)

        r_grid = r_coords.view(1, 1, 1, H, 1).expand(B, P, J, H, W)
        r_grid_flat = r_grid.reshape(B, P, J, H * W)

        az_grid = az_coords.view(1, 1, 1, 1, W).expand(B, P, J, H, W)
        az_grid_flat = az_grid.reshape(B, P, J, H * W)

        heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        r_pred = (r_grid_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum

        sin_weighted = (torch.sin(az_grid_flat) * heatmaps_flat).sum(dim=-1)
        cos_weighted = (torch.cos(az_grid_flat) * heatmaps_flat).sum(dim=-1)
        az_pred = torch.atan2(sin_weighted, cos_weighted).unsqueeze(-1)

        el_pred = (elevation_flat * heatmaps_flat).sum(
            dim=-1, keepdim=True
        ) / heatmap_sum

        el_pred = (
            el_pred * (self.el_max - self.el_min) / 2.0
            + (self.el_max + self.el_min) / 2.0
        )

        keypoints = torch.cat([r_pred, az_pred, el_pred], dim=-1)

        if squeeze_batch:
            keypoints = keypoints.squeeze(0)

        return keypoints

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()

        total_mpjpe = 0.0
        total_persons = 0
        joint_errors = []
        sample_summaries = []
        sample_payloads = []
        per_person_mpjpes = []

        for result in results:
            pred = result["pred_keypoints"]
            gt = result["gt_keypoints"]
            sample_name = result.get("sample_name", "unknown")

            pred, pred_mask = self._sanitize_joints(pred, treat_zero_as_invalid=False)
            gt, gt_mask = self._sanitize_joints(gt, treat_zero_as_invalid=True)

            pred_cart = spherical_to_cartesian(pred)
            gt_cart = spherical_to_cartesian(gt)

            mpjpe, matched_pairs, pair_errors, matches = self._match_and_compute_mpjpe(
                pred_cart, gt_cart, pred_mask, gt_mask
            )

            if mpjpe is not None:
                total_mpjpe += mpjpe * matched_pairs
                total_persons += matched_pairs
                per_person_mpjpes.extend(pair_errors)
                summary = dict(
                    sample_name=sample_name,
                    mpjpe=float(mpjpe),
                    matched_pairs=int(matched_pairs),
                )
                sample_summaries.append(summary)
                sample_payloads.append(
                    dict(
                        sample_name=sample_name,
                        mpjpe=float(mpjpe),
                        matched_pairs=int(matched_pairs),
                        people=self._build_matched_people_payload(
                            pred_sph=pred,
                            pred_cart=pred_cart,
                            gt_sph=gt,
                            gt_cart=gt_cart,
                            gt_mask=gt_mask,
                            matches=matches,
                        ),
                    )
                )

                joint_errors.extend(
                    self._compute_per_joint_errors(
                        pred_cart, gt_cart, pred_mask, gt_mask
                    )
                )

        if total_persons > 0:
            avg_mpjpe = total_mpjpe / total_persons
        else:
            avg_mpjpe = 0.0
            logger.warning("No valid person pairs found for MPJPE computation")

        metrics = {"MPJPE": avg_mpjpe, "num_persons": total_persons}

        if per_person_mpjpes:
            mpjpe_array = np.asarray(per_person_mpjpes, dtype=np.float64)
            metrics["MPJPE_median"] = float(np.median(mpjpe_array))
            metrics["MPJPE_p85"] = float(np.percentile(mpjpe_array, 85))
            metrics["MPJPE_p90"] = float(np.percentile(mpjpe_array, 90))
            metrics["MPJPE_p95"] = float(np.percentile(mpjpe_array, 95))
        else:
            metrics["MPJPE_median"] = 0.0
            metrics["MPJPE_p85"] = 0.0
            metrics["MPJPE_p90"] = 0.0
            metrics["MPJPE_p95"] = 0.0

        if joint_errors:
            joint_errors = torch.stack(joint_errors)
            for j in range(joint_errors.size(1)):
                metrics[f"MPJPE_joint_{j}"] = joint_errors[:, j].mean().item()

        self._dump_sample_analysis(sample_summaries, sample_payloads, metrics)

        logger.info(
            f"Evaluation complete: MPJPE = {avg_mpjpe:.4f} meters over {total_persons} persons"
        )

        return metrics

    def _dump_sample_analysis(
        self,
        sample_summaries: List[Dict[str, object]],
        sample_payloads: List[Dict[str, object]],
        metrics: Dict[str, float],
    ) -> None:
        if self.analysis_out_dir is None:
            return

        self.analysis_out_dir.mkdir(parents=True, exist_ok=True)
        low_dir = self.analysis_out_dir / f"mpjpe_lt_{self.low_mpjpe_thr:.3f}"
        high_dir = self.analysis_out_dir / f"mpjpe_gt_{self.high_mpjpe_thr:.3f}"
        low_dir.mkdir(parents=True, exist_ok=True)
        high_dir.mkdir(parents=True, exist_ok=True)

        for payload in sample_payloads:
            sample_name = str(payload["sample_name"])
            mpjpe = float(payload["mpjpe"])
            if mpjpe < self.low_mpjpe_thr:
                out_path = low_dir / f"{sample_name}.json"
            elif mpjpe > self.high_mpjpe_thr:
                out_path = high_dir / f"{sample_name}.json"
            else:
                continue

            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(payload["people"], handle, ensure_ascii=False, indent=2)

        summary_path = self.analysis_out_dir / "mpjpe_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(
                    thresholds=dict(low=self.low_mpjpe_thr, high=self.high_mpjpe_thr),
                    dataset_metrics=metrics,
                    samples=sample_summaries,
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _bbox_from_points(points: torch.Tensor) -> List[float]:
        if points.numel() == 0:
            return [0.0] * 6
        mins, _ = points.min(dim=0)
        maxs, _ = points.max(dim=0)
        center = (maxs + mins) / 2
        size = maxs - mins
        return [
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(size[0]),
            float(size[1]),
            float(size[2]),
        ]

    def _build_matched_people_payload(
        self,
        pred_sph: torch.Tensor,
        pred_cart: torch.Tensor,
        gt_sph: torch.Tensor,
        gt_cart: torch.Tensor,
        gt_mask: torch.Tensor,
        matches: List[Tuple[int, int]],
    ) -> List[Dict[str, object]]:
        people: List[Dict[str, object]] = []
        for pred_idx, gt_idx in matches:
            pred_cart_matched = pred_cart[pred_idx].detach().cpu()
            pred_sph_matched = pred_sph[pred_idx].detach().cpu()
            gt_cart_matched = gt_cart[gt_idx].detach().cpu()
            gt_sph_matched = gt_sph[gt_idx].detach().cpu()
            joint_mask = gt_mask[gt_idx].detach().cpu()

            pred_kp3d = torch.cat(
                [pred_cart_matched, torch.ones(pred_cart_matched.size(0), 1)], dim=-1
            ).tolist()
            pred_kpsp = torch.cat(
                [pred_sph_matched, torch.ones(pred_sph_matched.size(0), 1)], dim=-1
            ).tolist()
            gt_kp3d = torch.cat(
                [gt_cart_matched, torch.ones(gt_cart_matched.size(0), 1)], dim=-1
            ).tolist()
            gt_kpsp = torch.cat(
                [gt_sph_matched, torch.ones(gt_sph_matched.size(0), 1)], dim=-1
            ).tolist()

            errors = []
            for j in range(pred_cart_matched.size(0)):
                if bool(joint_mask[j]):
                    diff = pred_cart_matched[j] - gt_cart_matched[j]
                    errors.append(float(torch.norm(diff).item()))
                else:
                    errors.append(-1.0)

            valid_errors = [e for e in errors if e >= 0]
            person = dict(
                keypoints_3d=pred_kp3d,
                keypoints_sp=pred_kpsp,
                bbox_3d=self._bbox_from_points(pred_cart_matched),
                gt_keypoints_3d=gt_kp3d,
                gt_keypoints_sp=gt_kpsp,
                errors=errors,
            )
            if valid_errors:
                person["mean_error"] = float(np.mean(valid_errors))
                person["max_error"] = float(np.max(valid_errors))
            people.append(person)
        return people

    def _sanitize_joints(
        self, joints: torch.Tensor, treat_zero_as_invalid: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if joints.dim() == 2:
            joints = joints.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        finite_mask = ~(
            torch.isnan(joints).any(dim=-1) | torch.isinf(joints).any(dim=-1)
        )
        if treat_zero_as_invalid:
            non_zero_mask = joints.abs().sum(dim=-1) > 1e-8
            mask = finite_mask & non_zero_mask
        else:
            mask = finite_mask

        sanitized = torch.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)

        if squeeze_output:
            sanitized = sanitized.squeeze(0)
            mask = mask.squeeze(0)

        return sanitized, mask

    def _match_and_compute_mpjpe(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Tuple[Optional[float], int, List[float], List[Tuple[int, int]]]:
        num_pred = pred.size(0)
        gt.size(0)

        gt_person_valid = gt_mask.any(dim=1)
        valid_gt_indices = torch.where(gt_person_valid)[0].tolist()

        if len(valid_gt_indices) == 0:
            return None, 0, [], []

        num_valid_gt = len(valid_gt_indices)
        cost_matrix = torch.zeros(num_pred, num_valid_gt)

        for pi in range(num_pred):
            for gi, gt_idx in enumerate(valid_gt_indices):
                joint_mask = gt_mask[gt_idx] & pred_mask[pi]

                if joint_mask.any():
                    diff = pred[pi, joint_mask] - gt[gt_idx, joint_mask]
                    cost_matrix[pi, gi] = torch.norm(diff, dim=-1).mean()
                else:
                    cost_matrix[pi, gi] = 1e6

        best_assignment = self._find_optimal_assignment(cost_matrix, num_valid_gt)

        total_mpjpe = 0.0
        num_matched = 0
        pair_errors: List[float] = []
        matches: List[Tuple[int, int]] = []

        for gi, pi in enumerate(best_assignment):
            gt_idx = valid_gt_indices[gi]
            joint_mask = gt_mask[gt_idx] & pred_mask[pi]

            if joint_mask.any():
                diff = pred[pi, joint_mask] - gt[gt_idx, joint_mask]
                mpjpe = torch.norm(diff, dim=-1).mean()
                total_mpjpe += mpjpe.item()
                pair_errors.append(float(mpjpe.item()))
                num_matched += 1
                matches.append((int(pi), int(gt_idx)))

        if num_matched > 0:
            avg_mpjpe = total_mpjpe / num_matched
        else:
            avg_mpjpe = None

        return avg_mpjpe, num_matched, pair_errors, matches

    def _find_optimal_assignment(
        self, cost_matrix: torch.Tensor, num_gt: int
    ) -> List[int]:
        num_pred = cost_matrix.size(0)

        if num_gt == 0:
            return []

        if HAS_SCIPY and num_gt > 2:
            cost_np = cost_matrix.cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost_np)

            assignment = [-1] * num_gt
            for pred_idx, gt_idx in zip(row_ind, col_ind):
                if gt_idx < num_gt:
                    assignment[gt_idx] = int(pred_idx)

            for gi in range(num_gt):
                if assignment[gi] == -1:
                    used = set(assignment) - {-1}
                    best_pred = None
                    best_cost = float("inf")
                    for p in range(num_pred):
                        if p not in used and cost_matrix[p, gi] < best_cost:
                            best_cost = cost_matrix[p, gi]
                            best_pred = p
                    assignment[gi] = best_pred if best_pred is not None else 0

            return assignment

        if num_gt == 1:
            best_pred = cost_matrix[:, 0].argmin().item()
            return [best_pred]

        if num_gt == 2 and num_pred >= 2:
            min_cost = float("inf")
            best_perm = None

            for perm in permutations(range(num_pred), num_gt):
                cost = sum(cost_matrix[perm[i], i] for i in range(num_gt))
                if cost < min_cost:
                    min_cost = cost
                    best_perm = list(perm)

            return best_perm if best_perm else list(range(min(num_pred, num_gt)))

        used_preds = set()
        assignment = []

        for gi in range(num_gt):
            best_pred = None
            best_cost = float("inf")

            for pi in range(num_pred):
                if pi in used_preds:
                    continue
                if cost_matrix[pi, gi] < best_cost:
                    best_cost = cost_matrix[pi, gi]
                    best_pred = pi

            if best_pred is not None:
                assignment.append(best_pred)
                used_preds.add(best_pred)

        return assignment

    def _compute_per_joint_errors(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        num_pred = pred.size(0)
        num_joints = pred.size(1)

        gt_person_valid = gt_mask.any(dim=1)
        valid_gt_indices = torch.where(gt_person_valid)[0].tolist()

        if len(valid_gt_indices) == 0:
            return []

        num_valid_gt = len(valid_gt_indices)
        cost_matrix = torch.zeros(num_pred, num_valid_gt)

        for pi in range(num_pred):
            for gi, gt_idx in enumerate(valid_gt_indices):
                joint_mask = gt_mask[gt_idx] & pred_mask[pi]
                if joint_mask.any():
                    diff = pred[pi, joint_mask] - gt[gt_idx, joint_mask]
                    cost_matrix[pi, gi] = torch.norm(diff, dim=-1).mean()
                else:
                    cost_matrix[pi, gi] = 1e6

        assignment = self._find_optimal_assignment(cost_matrix, num_valid_gt)

        joint_errors = []
        for gi, pi in enumerate(assignment):
            gt_idx = valid_gt_indices[gi]
            joint_mask = gt_mask[gt_idx] & pred_mask[pi]

            errors = torch.zeros(num_joints)
            for j in range(num_joints):
                if joint_mask[j]:
                    diff = pred[pi, j] - gt[gt_idx, j]
                    errors[j] = torch.norm(diff).item()
                else:
                    errors[j] = 0.0

            joint_errors.append(errors)

        return joint_errors


@METRICS.register_module()
class IKMPJPE(HeatmapMPJPE):
    default_prefix: Optional[str] = "ik"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        point_cloud_range: Sequence[float] = [0.12, -0.524, -0.524, 6.0, 2.618, 0.524],
        softargmax_temperature: float = 3.0,
        analysis_out_dir: Optional[str] = None,
        low_mpjpe_thr: float = 0.06,
        high_mpjpe_thr: float = 0.10,
    ) -> None:
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            decode_from_heatmap=False,
            softargmax_temperature=softargmax_temperature,
            point_cloud_range=point_cloud_range,
            analysis_out_dir=analysis_out_dir,
            low_mpjpe_thr=low_mpjpe_thr,
            high_mpjpe_thr=high_mpjpe_thr,
        )

    def _get_pred_keypoints(
        self, data_sample, sample_idx: int, logger: MMLogger
    ) -> Optional[torch.Tensor]:

        pred_keypoints = self._get_field(data_sample, "pred_ik_keypoints_3d")
        if pred_keypoints is None:
            pred_keypoints = self._get_field(data_sample, "ik_keypoints")
        if pred_keypoints is None:
            pred_keypoints = self._get_field(data_sample, "pred_keypoints_3d")
        if pred_keypoints is None:
            pred_keypoints = self._get_field(data_sample, "pred_keypoints")
        if pred_keypoints is None:
            pred_keypoints = self._get_field(data_sample, "keypoints")
        return pred_keypoints
