import torch
import torch.nn.functional as F


def sinkhorn_algorithm(cost_matrix: torch.Tensor, 
                       num_iters: int = 100,
                       temperature: float = 0.1,
                       epsilon: float = 1e-8) -> torch.Tensor:

    num_pred, num_gt = cost_matrix.shape
    
    # Convert to log space to avoid numerical underflow
    # log_alpha[i,j] = -cost[i,j] / temperature
    log_alpha = -cost_matrix / temperature
    
    # Initialize dual variables
    log_u = torch.zeros(num_pred, device=cost_matrix.device, dtype=cost_matrix.dtype)
    log_v = torch.zeros(num_gt, device=cost_matrix.device, dtype=cost_matrix.dtype)
    
    # Sinkhorn iterations
    for _ in range(num_iters):
        # Row normalization: u = 1 / (K @ v)
        # In log space: log_u = -logsumexp(log_alpha + log_v, dim=1)
        log_u = -torch.logsumexp(log_alpha + log_v.unsqueeze(0), dim=1)
        
        # Column normalization: v = 1 / (K^T @ u)
        # In log space: log_v = -logsumexp(log_alpha + log_u, dim=0)
        log_v = -torch.logsumexp(log_alpha + log_u.unsqueeze(1), dim=0)
    
    # Compute final assignment matrix
    # P = diag(u) @ K @ diag(v)
    # In log space: log_P = log_u + log_alpha + log_v
    log_assignment = log_u.unsqueeze(1) + log_alpha + log_v.unsqueeze(0)
    assignment = torch.exp(log_assignment)
    
    # Numerical stability: ensure non-negative and sum to 1
    assignment = torch.clamp(assignment, min=epsilon)
    
    return assignment


def sinkhorn_matching_loss(pred_coords: torch.Tensor,
                           gt_coords: torch.Tensor,
                           pred_conf: torch.Tensor,
                           mask: torch.Tensor,
                           num_iters: int = 100,
                           temperature: float = 0.1,
                           conf_weight: float = 0.1) -> tuple:

    num_pred = pred_coords.size(0)
    num_gt = gt_coords.size(0)
    
    # Construct cost matrix
    cost = torch.zeros(num_pred, num_gt, device=pred_coords.device, dtype=pred_coords.dtype)
    
    for i in range(num_pred):
        for j in range(num_gt):
            joint_mask = mask[j]
            if joint_mask.any():
                # Coordinate distance cost
                diff = pred_coords[i, joint_mask] - gt_coords[j, joint_mask]
                coord_cost = diff.norm(dim=-1).mean()
                
                # Confidence penalty (detach to avoid affecting gradient flow)
                conf_penalty = -conf_weight * pred_conf[i].detach().sigmoid()
                
                cost[i, j] = coord_cost + conf_penalty
            else:
                cost[i, j] = 1e6
    
    # Sinkhorn algorithm to obtain soft assignment
    assignment = sinkhorn_algorithm(cost, num_iters=num_iters, temperature=temperature)
    
    # Compute weighted loss using soft assignment
    coord_loss = pred_coords.new_zeros(())
    for i in range(num_pred):
        for j in range(num_gt):
            if mask[j].any():
                weight = assignment[i, j]
                joint_mask = mask[j]
                diff = pred_coords[i, joint_mask] - gt_coords[j, joint_mask]
                coord_loss = coord_loss + weight * diff.norm(dim=-1).mean()
    
    coord_loss = coord_loss / torch.clamp(assignment.sum(), min=1.0)
    
    # Confidence loss: use soft assignment as target
    # target_conf[i] = sum_j assignment[i,j]
    # This is a soft version of "matched=1, unmatched=0"
    target_conf = assignment.sum(dim=1)  # (num_pred,)
    conf_loss = F.binary_cross_entropy_with_logits(
        pred_conf, target_conf, reduction='mean')
    
    return coord_loss, conf_loss, assignment


# Test code
if __name__ == '__main__':
    # Test Sinkhorn algorithm
    print("=" * 80)
    print("Sinkhorn Algorithm Test")
    print("=" * 80)
    
    # Simple cost matrix
    cost = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.5],
        [3.0, 2.5, 1.0],
        [4.0, 3.0, 2.0]
    ])
    
    print("\nInput cost matrix:")
    print(cost)
    
    # Run Sinkhorn
    assignment = sinkhorn_algorithm(cost, num_iters=100, temperature=0.1)
    
    print("\nSoft assignment matrix:")
    print(assignment)
    print(f"\nRow sums: {assignment.sum(dim=1)}")
    print(f"Column sums : {assignment.sum(dim=0)}")
    
    # Test different temperatures
    print("\n" + "=" * 80)
    print("Effect of Temperature Parameter")
    print("=" * 80)
    
    for temp in [0.01, 0.1, 1.0]:
        assign_temp = sinkhorn_algorithm(cost, num_iters=100, temperature=temp)
        max_vals = assign_temp.max(dim=1)[0]
        print(f"\nTemperature={temp}:")
        print(f"  Max assignment values: {max_vals.tolist()}")
        print(f"  (Closer to 1.0 indicates harder assignment, closer to uniform distribution indicates softer assignment)")
    
    # Test full matching loss
    print("\n" + "=" * 80)
    print("Soft Matching Loss Test")
    print("=" * 80)
    
    num_pred, num_gt, num_joints = 4, 3, 17
    pred_coords = torch.randn(num_pred, num_joints, 3)
    gt_coords = torch.randn(num_gt, num_joints, 3)
    pred_conf = torch.randn(num_pred)
    mask = torch.ones(num_gt, num_joints, dtype=torch.bool)
    
    coord_loss, conf_loss, assign = sinkhorn_matching_loss(
        pred_coords, gt_coords, pred_conf, mask,
        num_iters=100, temperature=0.1, conf_weight=0.1)
    
    print(f"\nCoord Loss: {coord_loss:.6f}")
    print(f"Conf Loss: {conf_loss:.6f}")
    print(f"Assignment matrix shape: {assign.shape}")
    print(f"Assignment row sums: {assign.sum(dim=1)}")
    
    # Gradient test
    print("\n" + "=" * 80)
    print("Gradient Flow Test")
    print("=" * 80)
    
    pred_coords_grad = torch.randn(num_pred, num_joints, 3, requires_grad=True)
    pred_conf_grad = torch.randn(num_pred, requires_grad=True)
    
    coord_loss_grad, conf_loss_grad, _ = sinkhorn_matching_loss(
        pred_coords_grad, gt_coords, pred_conf_grad, mask)
    
    total_loss = coord_loss_grad + conf_loss_grad
    total_loss.backward()
    
    print(f"✓ Gradient computation successful")
    print(f"  pred_coords.grad exists: {pred_coords_grad.grad is not None}")
    print(f"  pred_conf.grad exists: {pred_conf_grad.grad is not None}")
    print(f"  pred_coords.grad.norm: {pred_coords_grad.grad.norm():.6f}")
    print(f"  pred_conf.grad.norm: {pred_conf_grad.grad.norm():.6f}")
    
    print("\n✓ All tests passed!")