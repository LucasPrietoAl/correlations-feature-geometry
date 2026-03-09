import torch
import numpy as np
import math
from scipy.special import expit
from typing import Tuple
import argparse


def generate_spherical_bow_dataset(
    num_samples: int,
    num_features: int = 12,
    feature_sharpness: float = 5.0,
    feature_base_prob_log_odds: float = -2.0,
    assignment_noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.FloatTensor, float]:

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Generate feature vectors distributed evenly on a 3D sphere using a Fibonacci lattice
    indices = np.arange(0, num_features, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_features)
    theta = np.pi * (1 + 5**0.5) * indices
    W_features = np.column_stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ]).astype(np.float32)

    data = np.zeros((num_samples, num_features), dtype=np.float32)

    for i in range(num_samples):
        # Generate a random latent vector on the unit sphere for each sample
        u, v = rng.rand(2)
        lat_theta = 2 * math.pi * u
        lat_phi = np.arccos(2 * v - 1)
        latent_vec = np.array([
            math.sin(lat_phi) * math.cos(lat_theta),
            math.sin(lat_phi) * math.sin(lat_theta),
            math.cos(lat_phi)
        ], dtype=np.float32)

        if assignment_noise_std > 0.0:
            latent_vec += rng.normal(0.0, assignment_noise_std, size=3)
            latent_vec /= np.linalg.norm(latent_vec)

        # Calculate feature probabilities based on dot product
        logits = feature_sharpness * (W_features @ latent_vec) + feature_base_prob_log_odds
        probs = expit(logits)
        data[i] = rng.rand(num_features) < probs

    sparsity = data.mean().item()
    return torch.tensor(data), sparsity

def generate_figure8_bow_dataset(
    num_samples: int,
    num_features: int = 12,
    feature_sharpness: float = 5.0,
    feature_base_prob_log_odds: float = -2.0,
    assignment_noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.FloatTensor, float]:
    """
    Generates a dataset with a figure-8 covariance structure.
    Projects to a Lissajous curve (sin(t), sin(2t)) in the first two PCs.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, num_features, endpoint=False)
    W_features = np.column_stack((np.sin(angles), np.sin(2 * angles)))

    data = np.zeros((num_samples, num_features), dtype=np.float32)
    for i in range(num_samples):
        latent_pos = i % num_features
        theta = 2.0 * math.pi * latent_pos / num_features
        if assignment_noise_std > 0.0:
            theta += rng.normal(0.0, assignment_noise_std)
        latent_vec = np.array([math.sin(theta), math.sin(2 * theta)], dtype=np.float32)
        logits = feature_sharpness * (W_features @ latent_vec) + feature_base_prob_log_odds
        probs = expit(logits)
        data[i] = rng.rand(num_features) < probs

    sparsity = data.mean().item()
    return torch.tensor(data), sparsity

def generate_circular_bow_dataset(
    num_samples: int,
    num_features: int = 12,
    feature_sharpness: float = 5.0,
    feature_base_prob_log_odds: float = -2.0,
    assignment_noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.FloatTensor, float]:
    """
    Generates a dataset with a circular covariance structure (like months in a year).
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, num_features, endpoint=False)
    W_features = np.column_stack((np.cos(angles), np.sin(angles)))

    data = np.zeros((num_samples, num_features), dtype=np.float32)
    for i in range(num_samples):
        latent_pos = i % num_features
        theta = 2.0 * math.pi * latent_pos / num_features
        if assignment_noise_std > 0.0:
            theta += rng.normal(0.0, assignment_noise_std)
        latent_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        logits = feature_sharpness * (W_features @ latent_vec) + feature_base_prob_log_odds
        probs = expit(logits)
        data[i] = rng.rand(num_features) < probs
        
    sparsity = data.mean().item()
    return torch.tensor(data), sparsity

def _get_correlated_data(data_type: str, args: argparse.Namespace) -> Tuple[torch.Tensor, float]:
    common_params = {
        "num_samples": args.num_samples,
        "num_features": args.num_features,
        "feature_sharpness": args.feature_sharpness,
        "feature_base_prob_log_odds": args.feature_base_log_odds,
        "assignment_noise_std": args.feature_noise,
        "seed": args.data_seed,
    }
    if data_type == "circular":
        return generate_circular_bow_dataset(**common_params)
    elif data_type == "figure8":
        return generate_figure8_bow_dataset(**common_params)
    elif data_type == "sphere":
        return generate_spherical_bow_dataset(**common_params)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def _iid_data(num_samples: int, p: float, seed: int, num_features: int) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.rand(num_samples, num_features) < p, dtype=torch.float32)
