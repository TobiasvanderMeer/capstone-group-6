import torch
import torch.nn.functional as F

def darcy_physics_informed_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    k: torch.Tensor,
    h_mean: float = 146,
    h_std: float = 37,
    lambda_darcy: float = 0.01,
    lambda_bc: float = 1.0,
) -> torch.Tensor:
    """
    Fixed physics-informed loss for Darcy flow (axes aligned with H=rows, W=cols)
    """

    # 1. Data loss (normalized)
    loss_data = F.mse_loss(y_pred, y_true)

    # ------------------------
    # 2. Darcy PDE residual
    # ------------------------
    B, H, W = y_pred.shape
    h = y_pred.unsqueeze(1)  # (B,1,H,W)
    device = h.device

    # Finite difference kernels for central differences
    dx_kernel = torch.tensor(
        [[0, 0, 0],
         [-0.5, 0, 0.5],
         [0, 0, 0]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    dy_kernel = torch.tensor(
        [[0, -0.5, 0],
         [0, 0, 0],
         [0, 0.5, 0]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    # Gradients (normalized)
    dh_dx = F.conv2d(h, dx_kernel, padding=1)
    dh_dy = F.conv2d(h, dy_kernel, padding=1)

    # Convert to physical units
    dh_dx = h_std * dh_dx
    dh_dy = h_std * dh_dy

    # Face-averaged permeability
    k_center = k
    k_pad_x = F.pad(k_center, (0, 1, 0, 0), mode="replicate")
    k_dx = 0.5 * (k_center + k_pad_x[:, :, :, 1:])

    k_pad_y = F.pad(k_center, (0, 0, 0, 1), mode="replicate")
    k_dy = 0.5 * (k_center + k_pad_y[:, :, 1:, :])

    # Fluxes
    flux_x = k_dx * dh_dx
    flux_y = k_dy * dh_dy

    # Divergence
    div_x = F.conv2d(flux_x, dx_kernel, padding=1)
    div_y = F.conv2d(flux_y, dy_kernel, padding=1)
    div = div_x + div_y

    loss_darcy = torch.mean(div ** 2)


    # 3. Boundary condition losses

    eu = 1.0
    dx = 6.0 / 59.0
    dy = 6.0 / 59.0

    # Top Neumann BC dh/dy = 0
    top_penalty = torch.mean(((y_pred[:, 0, :] - y_pred[:, 1, :]) / dy) ** 2)
    # top_penalty = torch.mean((k[:, 0, :] * (y_pred[:, 0, :] - y_pred[:, 1, :]) / dy) ** 2)

    # Right Neumann BC dh/dx = 0
    right_penalty = torch.mean(((y_pred[:, :, -1] - y_pred[:, :, -2]) / dx) ** 2)

    # Left flux BC: -K dh/dx = 500
    left_flux_norm = 500.0 / h_std
    left_penalty = torch.mean((-eu * (y_pred[:, :, 1] - y_pred[:, :, 0]) / dx - left_flux_norm) ** 2)

    # Bottom Dirichlet BC h = 100
    bottom_bc = (100.0 - h_mean) / h_std
    bottom_penalty = torch.mean((y_pred[:, -1, :] - bottom_bc) ** 2)



    loss_bc = (
        0.5 * top_penalty +
        1 * right_penalty +
        0.5 * left_penalty +
        1 * bottom_penalty)


    total_loss = loss_data + lambda_bc * loss_bc + lambda_darcy * loss_darcy

    return total_loss