import torch


def spherical_to_cartesian(coords: torch.Tensor) -> torch.Tensor:
    if coords is None or coords.shape[-1] < 3:
        return coords
    r = coords[..., 0]
    az = coords[..., 1]
    el = coords[..., 2]
    cos_el = torch.cos(el)
    x = r * cos_el * torch.cos(az)
    y = r * torch.sin(el)
    z = r * cos_el * torch.sin(az)
    xyz = torch.stack((x, y, z), dim=-1)
    return torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
