import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional


def load_raw_dense_reliability_matrix(file_path: str) -> np.ndarray:
    data = pd.read_csv(file_path, header=None).to_numpy()

    final_data = []

    for i in range(len(data)):
        temp = data[i, 0].split("\t")
        ans = []
        for j in range(len(temp)):
            if temp[j] == "":
                continue

            val = float(temp[j])
            if val == -1:
                ans.append(0.0)
            else:
                ans.append(val)
        final_data.append(ans)

    final_data = np.array(final_data)
    return final_data


def load_all_raw_dense_metrics(dataset_path: str, param_suffix: str) -> Dict[str, np.ndarray]:
    metric_names = ["SuccessRate", "rightBlock", "recentHeight", "roundtripTime"]
    metric_keys = ["SR", "RB", "RH", "RTT"]

    raw_metrics_dict = {}

    for metric_name, metric_key in zip(metric_names, metric_keys):
        file_path = os.path.join(dataset_path, f"{metric_name}{param_suffix}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metric file not found: {file_path}")

        raw_matrix = load_raw_dense_reliability_matrix(file_path)
        raw_metrics_dict[metric_key] = raw_matrix

    return raw_metrics_dict


def split_and_normalize_dense_data(
    raw_metrics_dict: Dict[str, np.ndarray], args
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    train_dense_norm_dict = {}
    valid_dense_norm_dict = {}
    test_dense_norm_dict = {}
    scale_factors_dict = {}

    primary_metric = "SR"
    if primary_metric not in raw_metrics_dict:
        raise ValueError(f"Primary metric {primary_metric} is missing from the input data")

    num_users, num_services = raw_metrics_dict[primary_metric].shape

    all_possible_indices = []
    for r_idx in range(num_users):
        for c_idx in range(num_services):
            all_possible_indices.append((r_idx, c_idx))
    all_possible_indices = np.array(all_possible_indices)

    total_samples = len(all_possible_indices)

    train_ratio = args.train_density
    valid_ratio = args.valid_density

    train_size = int(total_samples * train_ratio)
    valid_size = int(total_samples * valid_ratio)
    test_size = total_samples - train_size - valid_size

    if test_size < 0:
        raise ValueError(
            "The training and validation ratios exceed 1, leaving a negative test size"
        )

    np.random.seed(getattr(args, "random_seed", 42))
    shuffled_master_indices = np.random.permutation(total_samples)

    train_indices_flat = shuffled_master_indices[:train_size]
    valid_indices_flat = shuffled_master_indices[train_size : train_size + valid_size]
    test_indices_flat = shuffled_master_indices[train_size + valid_size :]

    train_row_indices, train_col_indices = (
        all_possible_indices[train_indices_flat, 0],
        all_possible_indices[train_indices_flat, 1],
    )
    valid_row_indices, valid_col_indices = (
        all_possible_indices[valid_indices_flat, 0],
        all_possible_indices[valid_indices_flat, 1],
    )
    test_row_indices, test_col_indices = (
        all_possible_indices[test_indices_flat, 0],
        all_possible_indices[test_indices_flat, 1],
    )

    for metric_key, raw_matrix in raw_metrics_dict.items():
        tensor = raw_matrix.copy().astype(float)

        if metric_key == "RTT":
            tensor = np.log1p(tensor)

        if metric_key == "RTT":
            train_values = tensor[train_row_indices, train_col_indices]
            train_max = np.max(train_values) if train_values.size > 0 else 0.0
            scale_factor = train_max if train_max > 0 else 1.0
            tensor = tensor / scale_factor
            scale_factors_dict[metric_key] = scale_factor
        else:
            scale_factors_dict[metric_key] = 1.0

        train_dense_normalized = np.zeros_like(tensor)
        valid_dense_normalized = np.zeros_like(tensor)
        test_dense_normalized = np.zeros_like(tensor)

        train_dense_normalized[train_row_indices, train_col_indices] = tensor[
            train_row_indices, train_col_indices
        ]
        valid_dense_normalized[valid_row_indices, valid_col_indices] = tensor[
            valid_row_indices, valid_col_indices
        ]
        test_dense_normalized[test_row_indices, test_col_indices] = tensor[
            test_row_indices, test_col_indices
        ]

        train_dense_norm_dict[metric_key] = train_dense_normalized
        valid_dense_norm_dict[metric_key] = valid_dense_normalized
        test_dense_norm_dict[metric_key] = test_dense_normalized

    num_train_samples = len(train_row_indices)
    num_valid_samples = len(valid_row_indices)
    num_test_samples = len(test_row_indices)
    total_allocated_samples = num_train_samples + num_valid_samples + num_test_samples

    return train_dense_norm_dict, valid_dense_norm_dict, test_dense_norm_dict, scale_factors_dict


class BlockchainReliabilityDataset(Dataset):
    def __init__(self, metrics_data_dict_for_this_split: Dict[str, np.ndarray], args):
        self.metrics_data_dict = metrics_data_dict_for_this_split
        self.args = args

        self.interaction_data = self._generate_interaction_data_from_dense_multi_target()

    def _generate_interaction_data_from_dense_multi_target(
        self,
    ) -> List[Tuple[int, int, float, float, float, float]]:
        primary_metric = "SR"
        primary_matrix = self.metrics_data_dict[primary_metric]

        user_indices, service_indices = primary_matrix.nonzero()

        interactions = []

        for u, s in zip(user_indices, service_indices):
            val_SR = self.metrics_data_dict["SR"][u, s]
            val_RB = self.metrics_data_dict["RB"][u, s]
            val_RH = self.metrics_data_dict["RH"][u, s]
            val_RTT = self.metrics_data_dict["RTT"][u, s]

            interactions.append((u, s, val_SR, val_RB, val_RH, val_RTT))

        return interactions

    def __len__(self):
        return len(self.interaction_data)

    def __getitem__(self, index):
        user_idx, serv_idx, val_SR, val_RB, val_RH, val_RTT = self.interaction_data[index]

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(serv_idx, dtype=torch.long),
            torch.tensor(val_SR, dtype=torch.float32),
            torch.tensor(val_RB, dtype=torch.float32),
            torch.tensor(val_RH, dtype=torch.float32),
            torch.tensor(val_RTT, dtype=torch.float32),
        )


def get_dataloaders_for_reliability(
    args, param_suffix: str = "_12_1000", batch_size: int = 128, random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    dataset_path = getattr(args, "dataset_path", "dataset/")

    raw_metrics_dict = load_all_raw_dense_metrics(dataset_path, param_suffix)

    shapes = [matrix.shape for matrix in raw_metrics_dict.values()]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All metric matrices must have the same shape")

    (
        train_dense_norm_dict,
        valid_dense_norm_dict,
        test_dense_norm_dict,
        scale_factors_dict,
    ) = split_and_normalize_dense_data(raw_metrics_dict, args)

    train_interactions = np.count_nonzero(train_dense_norm_dict["SR"])
    valid_interactions = np.count_nonzero(valid_dense_norm_dict["SR"])
    test_interactions = np.count_nonzero(test_dense_norm_dict["SR"])
    total_interactions = train_interactions + valid_interactions + test_interactions

    train_dataset = BlockchainReliabilityDataset(train_dense_norm_dict, args)
    valid_dataset = BlockchainReliabilityDataset(valid_dense_norm_dict, args)
    test_dataset = BlockchainReliabilityDataset(test_dense_norm_dict, args)

    num_workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    normalization_params = {
        "scale_factors": scale_factors_dict,
        "num_users": shapes[0][0],
        "num_services": shapes[0][1],
    }

    return train_loader, valid_loader, test_loader, normalization_params


def denormalize_predictions(
    predictions: torch.Tensor, normalization_params: Dict, metric_name: str
) -> torch.Tensor:
    if "scale_factors" not in normalization_params:
        raise ValueError("The normalization parameters do not contain 'scale_factors'")

    if metric_name not in normalization_params["scale_factors"]:
        raise ValueError(f"Metric '{metric_name}' is missing from scale_factors")

    scale_factor = normalization_params["scale_factors"][metric_name]
    return predictions * scale_factor
