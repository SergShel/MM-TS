import torch

def compute_cluster_based_shift(
    clusters: torch.Tensor,
    min_cluster_size: int,
    max_cluster_size: int,
    min_shift: float,
    max_shift: float,
) -> float:
    """Compute the shift for MM-TS based on the cluster size of the current sample."""
    return (clusters - min_cluster_size) / (max_cluster_size - min_cluster_size) * (
        max_shift - min_shift
    ) + min_shift


if __name__ == "__main__":

    # Example usage
    clusters = torch.tensor([10.0, 12.0, 14.0, 6.0])
    min_cluster_size = 1
    max_cluster_size = 20
    min_shift = 0.05
    max_shift = 0.20

    shift = compute_cluster_based_shift(
        clusters,
        min_cluster_size,
        max_cluster_size,
        min_shift,
        max_shift,
    )
    print(f"Computed shift: {shift}")