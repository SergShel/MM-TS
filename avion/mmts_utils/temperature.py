import torch


def compute_tau_base(global_step: int, alpha: float, number_steps_per_period: int) -> float:
    """Compute the base temperature (tau) for MM-TS based on the current training step."""

    # Make sure global_step and period are tensors
    global_step = torch.tensor(global_step, dtype=torch.float32)
    number_steps_per_period = torch.tensor(number_steps_per_period, dtype=torch.float32)

    return (
        0.5 * (alpha * (1.0 + torch.cos(2.0 * torch.pi * global_step / number_steps_per_period)) - alpha)
    )


if __name__ == "__main__":
    # Example usage
    for step in range(0, 10000, 1000):
        tau = compute_tau_base(global_step=step, alpha=0.08, number_steps_per_period=10000)
        print(f"Step: {step}, Tau Base: {tau:.4f}")
