"""
Visualization tool for 2.5D heatmap-based pose estimation.

This tool visualizes:
1. RA (Range-Azimuth) heatmaps for each joint
2. Elevation maps overlaid on heatmaps
3. Decoded 3D keypoints projected on heatmap
4. Ground truth vs prediction comparison
5. Multi-person visualization with different colors

Usage:
    python tools/visualize_heatmap.py \
        --config configs/mmwave_heatmap_2person.py \
        --checkpoint work_dirs/mmwave_heatmap_2person/best.pth \
        --data-root singleperson/val \
        --out-dir visualizations/heatmap \
        --show  # Optional: display instead of save
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine.config import Config
from mmengine.runner import Runner
from mmwave3d.utils import spherical_to_cartesian


class HeatmapVisualizer:
    """Visualizer for 2.5D heatmaps"""
    
    def __init__(self, config_file, checkpoint_file=None):
        """
        Initialize visualizer.
        
        Args:
            config_file: Path to config file
            checkpoint_file: Path to checkpoint (optional, for loading trained model)
        """
        self.cfg = Config.fromfile(config_file)
        self.checkpoint = checkpoint_file
        
        # Extract heatmap parameters from config
        head_cfg = self.cfg.model.head
        self.num_person = head_cfg.get('num_person', 2)
        self.num_joints = head_cfg.get('num_joints', 17)
        self.range_bins = head_cfg.get('range_bins', 50)
        self.angle_bins = head_cfg.get('angle_bins', 180)
        
        # Point cloud range
        pcr = head_cfg.get('point_cloud_range', [0.12, -0.524, -0.524, 6.0, 2.618, 0.524])
        self.r_min, self.az_min, self.el_min = pcr[:3]
        self.r_max, self.az_max, self.el_max = pcr[3:]
        
        # Joint names (COCO 17 keypoints)
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Color palette for different persons
        self.person_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    def visualize_single_heatmap(self, 
                                 heatmap, 
                                 elevation_map=None,
                                 pred_keypoint=None,
                                 gt_keypoint=None,
                                 joint_idx=0,
                                 person_idx=0,
                                 title=None,
                                 save_path=None,
                                 show=False):
        """
        Visualize a single joint's heatmap.
        
        Args:
            heatmap: (H, W) - RA heatmap
            elevation_map: (H, W) - elevation map (optional)
            pred_keypoint: (3,) - predicted keypoint in spherical coords
            gt_keypoint: (3,) - ground truth keypoint in spherical coords
            joint_idx: Joint index
            person_idx: Person index
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure
        """
        fig, axes = plt.subplots(1, 2 if elevation_map is not None else 1, 
                                figsize=(12 if elevation_map is not None else 6, 5))
        
        if elevation_map is None:
            axes = [axes]
        
        # Create coordinate grids
        r_coords = np.linspace(self.r_min, self.r_max, self.range_bins)
        az_coords = np.linspace(self.az_min, self.az_max, self.angle_bins)
        
        # Convert to numpy
        heatmap_np = heatmap.cpu().numpy() if torch.is_tensor(heatmap) else heatmap
        
        # Plot heatmap
        ax = axes[0]
        im = ax.imshow(heatmap_np, cmap='hot', aspect='auto', 
                      extent=[np.degrees(self.az_min), np.degrees(self.az_max), 
                             self.r_max, self.r_min],
                      interpolation='bilinear')
        
        ax.set_xlabel('Azimuth (degrees)', fontsize=12)
        ax.set_ylabel('Range (meters)', fontsize=12)
        ax.set_title(f'Heatmap - Person {person_idx}, {self.joint_names[joint_idx]}', 
                    fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Confidence')
        
        # Overlay predicted keypoint
        if pred_keypoint is not None:
            r_pred, az_pred, _ = pred_keypoint
            if torch.is_tensor(r_pred):
                r_pred = r_pred.item()
                az_pred = az_pred.item()
            ax.plot(np.degrees(az_pred), r_pred, 'c*', markersize=15, 
                   markeredgecolor='white', markeredgewidth=1.5, label='Predicted')
        
        # Overlay ground truth keypoint
        if gt_keypoint is not None:
            r_gt, az_gt, _ = gt_keypoint
            if torch.is_tensor(r_gt):
                r_gt = r_gt.item()
                az_gt = az_gt.item()
            ax.plot(np.degrees(az_gt), r_gt, 'g^', markersize=12, 
                   markeredgecolor='white', markeredgewidth=1.5, label='Ground Truth')
        
        if pred_keypoint is not None or gt_keypoint is not None:
            ax.legend(loc='upper right', fontsize=10)
        
        # Plot elevation map if provided
        if elevation_map is not None:
            ax = axes[1]
            elevation_np = elevation_map.cpu().numpy() if torch.is_tensor(elevation_map) else elevation_map
            
            # Mask by heatmap confidence
            masked_elevation = np.ma.masked_where(heatmap_np < 0.1, elevation_np)
            
            im = ax.imshow(masked_elevation, cmap='coolwarm', aspect='auto',
                          extent=[np.degrees(self.az_min), np.degrees(self.az_max), 
                                 self.r_max, self.r_min],
                          interpolation='bilinear', vmin=-1, vmax=1)
            
            ax.set_xlabel('Azimuth (degrees)', fontsize=12)
            ax.set_ylabel('Range (meters)', fontsize=12)
            ax.set_title(f'Elevation Map - Person {person_idx}, {self.joint_names[joint_idx]}', 
                        fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Elevation (normalized)')
            
            # Overlay keypoints
            if pred_keypoint is not None:
                r_pred, az_pred, el_pred = pred_keypoint
                if torch.is_tensor(r_pred):
                    r_pred = r_pred.item()
                    az_pred = az_pred.item()
                    el_pred = el_pred.item()
                ax.plot(np.degrees(az_pred), r_pred, 'c*', markersize=15, 
                       markeredgecolor='white', markeredgewidth=1.5)
                ax.text(np.degrees(az_pred), r_pred - 0.3, f'el={el_pred:.2f}',
                       color='cyan', fontsize=9, ha='center')
            
            if gt_keypoint is not None:
                r_gt, az_gt, el_gt = gt_keypoint
                if torch.is_tensor(r_gt):
                    r_gt = r_gt.item()
                    az_gt = az_gt.item()
                    el_gt = el_gt.item()
                ax.plot(np.degrees(az_gt), r_gt, 'g^', markersize=12, 
                       markeredgecolor='white', markeredgewidth=1.5)
                ax.text(np.degrees(az_gt), r_gt + 0.3, f'el={el_gt:.2f}',
                       color='lime', fontsize=9, ha='center')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_all_joints(self,
                            heatmaps,
                            elevation_maps=None,
                            pred_keypoints=None,
                            gt_keypoints=None,
                            person_idx=0,
                            save_dir=None,
                            show=False,
                            max_joints=None):
        """
        Visualize all joints for one person in a grid.
        
        Args:
            heatmaps: (num_joints, H, W)
            elevation_maps: (num_joints, H, W) optional
            pred_keypoints: (num_joints, 3)
            gt_keypoints: (num_joints, 3)
            person_idx: Person index
            save_dir: Directory to save individual joint visualizations
            show: Whether to display
            max_joints: Maximum number of joints to visualize
        """
        num_joints = min(heatmaps.shape[0], max_joints or self.num_joints)
        
        # Create grid
        cols = 4
        rows = (num_joints + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for j in range(num_joints):
            ax = axes[j]
            
            heatmap = heatmaps[j].cpu().numpy() if torch.is_tensor(heatmaps[j]) else heatmaps[j]
            
            im = ax.imshow(heatmap, cmap='hot', aspect='auto',
                          extent=[np.degrees(self.az_min), np.degrees(self.az_max),
                                 self.r_max, self.r_min],
                          interpolation='bilinear')
            
            ax.set_title(self.joint_names[j], fontsize=10, fontweight='bold')
            ax.set_xlabel('Azimuth (°)', fontsize=8)
            ax.set_ylabel('Range (m)', fontsize=8)
            
            # Overlay predictions
            if pred_keypoints is not None:
                r_pred, az_pred, _ = pred_keypoints[j]
                if torch.is_tensor(r_pred):
                    r_pred = r_pred.item()
                    az_pred = az_pred.item()
                ax.plot(np.degrees(az_pred), r_pred, 'c*', markersize=10,
                       markeredgecolor='white', markeredgewidth=1)
            
            if gt_keypoints is not None:
                r_gt, az_gt, _ = gt_keypoints[j]
                if torch.is_tensor(r_gt):
                    r_gt = r_gt.item()
                    az_gt = az_gt.item()
                ax.plot(np.degrees(az_gt), r_gt, 'g^', markersize=8,
                       markeredgecolor='white', markeredgewidth=1)
        
        # Hide unused subplots
        for j in range(num_joints, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(f'All Joints - Person {person_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'person_{person_idx}_all_joints.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_multi_person_skeleton(self,
                                       pred_keypoints_list,
                                       gt_keypoints_list=None,
                                       title='Multi-Person Pose',
                                       save_path=None,
                                       show=False):
        """
        Visualize 3D skeletons for multiple persons.
        
        Args:
            pred_keypoints_list: List of (num_joints, 3) for each person
            gt_keypoints_list: List of GT keypoints (optional)
            title: Plot title
            save_path: Path to save
            show: Whether to display
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Define skeleton connections (COCO format)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Top view (XY plane)
        ax1 = fig.add_subplot(131)
        ax1.set_title('Top View (XY)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Front view (XZ plane)
        ax2 = fig.add_subplot(132)
        ax2.set_title('Front View (XZ)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Side view (YZ plane)
        ax3 = fig.add_subplot(133)
        ax3.set_title('Side View (YZ)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Plot each person
        for person_idx, pred_kpts in enumerate(pred_keypoints_list):
            color = self.person_colors[person_idx % len(self.person_colors)]
            
            # Convert to Cartesian
            if torch.is_tensor(pred_kpts):
                pred_kpts = pred_kpts.cpu().numpy()
            
            # Assume input is spherical, convert to Cartesian
            if pred_kpts.shape[-1] == 3:
                # spherical_to_cartesian expects (r, az, el)
                pred_cart = []
                for joint in pred_kpts:
                    r, az, el = joint
                    x = r * np.cos(el) * np.cos(az)
                    y = r * np.cos(el) * np.sin(az)
                    z = r * np.sin(el)
                    pred_cart.append([x, y, z])
                pred_cart = np.array(pred_cart)
            else:
                pred_cart = pred_kpts
            
            # Plot joints
            ax1.scatter(pred_cart[:, 0], pred_cart[:, 1], c=color, s=50, 
                       label=f'Person {person_idx}', alpha=0.8, edgecolors='black')
            ax2.scatter(pred_cart[:, 0], pred_cart[:, 2], c=color, s=50, alpha=0.8, edgecolors='black')
            ax3.scatter(pred_cart[:, 1], pred_cart[:, 2], c=color, s=50, alpha=0.8, edgecolors='black')
            
            # Plot skeleton connections
            for start, end in skeleton:
                if start < len(pred_cart) and end < len(pred_cart):
                    ax1.plot([pred_cart[start, 0], pred_cart[end, 0]],
                            [pred_cart[start, 1], pred_cart[end, 1]],
                            c=color, linewidth=2, alpha=0.7)
                    ax2.plot([pred_cart[start, 0], pred_cart[end, 0]],
                            [pred_cart[start, 2], pred_cart[end, 2]],
                            c=color, linewidth=2, alpha=0.7)
                    ax3.plot([pred_cart[start, 1], pred_cart[end, 1]],
                            [pred_cart[start, 2], pred_cart[end, 2]],
                            c=color, linewidth=2, alpha=0.7)
        
        # Plot ground truth if provided
        if gt_keypoints_list is not None:
            for person_idx, gt_kpts in enumerate(gt_keypoints_list):
                if torch.is_tensor(gt_kpts):
                    gt_kpts = gt_kpts.cpu().numpy()
                
                # Convert to Cartesian
                gt_cart = []
                for joint in gt_kpts:
                    r, az, el = joint
                    x = r * np.cos(el) * np.cos(az)
                    y = r * np.cos(el) * np.sin(az)
                    z = r * np.sin(el)
                    gt_cart.append([x, y, z])
                gt_cart = np.array(gt_cart)
                
                # Plot with different marker
                ax1.scatter(gt_cart[:, 0], gt_cart[:, 1], c='lime', s=30, 
                           marker='^', alpha=0.6, edgecolors='black')
                ax2.scatter(gt_cart[:, 0], gt_cart[:, 2], c='lime', s=30, 
                           marker='^', alpha=0.6, edgecolors='black')
                ax3.scatter(gt_cart[:, 1], gt_cart[:, 2], c='lime', s=30, 
                           marker='^', alpha=0.6, edgecolors='black')
        
        ax1.legend(loc='best')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize heatmap predictions')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file path (optional)')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--joint-idx', type=int, default=None, 
                       help='Specific joint to visualize (default: all)')
    parser.add_argument('--person-idx', type=int, default=0, help='Person index')
    parser.add_argument('--out-dir', default='visualizations/heatmap', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Display instead of save')
    parser.add_argument('--data-root', help='Override data root from config')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create visualizer
    visualizer = HeatmapVisualizer(args.config, args.checkpoint)
    
    print(f"Visualizer initialized:")
    print(f"  - Number of persons: {visualizer.num_person}")
    print(f"  - Number of joints: {visualizer.num_joints}")
    print(f"  - Heatmap resolution: {visualizer.range_bins} x {visualizer.angle_bins}")
    
    # Example: Create synthetic data for demonstration
    print("\nGenerating example visualization...")
    
    # Create synthetic heatmap with a Gaussian peak
    heatmap = torch.zeros(visualizer.range_bins, visualizer.angle_bins)
    center_r, center_az = 25, 90
    for i in range(visualizer.range_bins):
        for j in range(visualizer.angle_bins):
            dist = ((i - center_r) ** 2 + (j - center_az) ** 2)
            heatmap[i, j] = np.exp(-dist / 50.0)
    
    # Create synthetic elevation map
    elevation_map = torch.randn(visualizer.range_bins, visualizer.angle_bins) * 0.3
    
    # Synthetic keypoints
    r_coords = torch.linspace(visualizer.r_min, visualizer.r_max, visualizer.range_bins)
    az_coords = torch.linspace(visualizer.az_min, visualizer.az_max, visualizer.angle_bins)
    pred_keypoint = torch.tensor([r_coords[center_r], az_coords[center_az], 0.1])
    gt_keypoint = torch.tensor([r_coords[center_r - 2], az_coords[center_az + 5], 0.15])
    
    # Visualize single joint
    joint_idx = args.joint_idx if args.joint_idx is not None else 5
    save_path = os.path.join(args.out_dir, f'joint_{joint_idx}_person_{args.person_idx}.png')
    
    visualizer.visualize_single_heatmap(
        heatmap=heatmap,
        elevation_map=elevation_map,
        pred_keypoint=pred_keypoint,
        gt_keypoint=gt_keypoint,
        joint_idx=joint_idx,
        person_idx=args.person_idx,
        title=f'Sample {args.sample_idx}',
        save_path=save_path if not args.show else None,
        show=args.show
    )
    
    # Visualize all joints
    if args.joint_idx is None:
        all_heatmaps = torch.stack([heatmap] * visualizer.num_joints)
        all_elevations = torch.stack([elevation_map] * visualizer.num_joints)
        pred_keypoints = torch.stack([pred_keypoint] * visualizer.num_joints)
        gt_keypoints = torch.stack([gt_keypoint] * visualizer.num_joints)
        
        visualizer.visualize_all_joints(
            heatmaps=all_heatmaps,
            elevation_maps=all_elevations,
            pred_keypoints=pred_keypoints,
            gt_keypoints=gt_keypoints,
            person_idx=args.person_idx,
            save_dir=args.out_dir if not args.show else None,
            show=args.show
        )
    
    # Visualize multi-person skeleton
    pred_keypoints_list = [
        torch.randn(17, 3) * 0.5 + torch.tensor([2.0, 0.5, 0.0]),
        torch.randn(17, 3) * 0.5 + torch.tensor([3.0, -0.3, 0.1])
    ]
    
    save_path = os.path.join(args.out_dir, 'multi_person_skeleton.png')
    visualizer.visualize_multi_person_skeleton(
        pred_keypoints_list=pred_keypoints_list,
        title=f'Multi-Person Skeleton - Sample {args.sample_idx}',
        save_path=save_path if not args.show else None,
        show=args.show
    )
    
    print(f"\n✓ Visualization complete!")
    if not args.show:
        print(f"Results saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
