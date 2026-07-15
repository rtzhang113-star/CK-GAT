import torch
import numpy as np
from time import time
import collections
import os
import copy


from config import parse_args
from core_utils.common_utils import (
    set_seed,
    to_cuda,
    optimizer_zero_grad,
    optimizer_step,
    lr_scheduler_step,
    result_append,
)
from core_utils.logger_utils import Logger
from core_utils.training_tools import get_loss_function, get_optimizer
from core_utils.early_stopping import EarlyStopping
from core_utils.evaluation_metrics import ErrMetrics
from data_processing.dataset_loader import get_dataloaders_for_reliability, denormalize_predictions
from graph_construction.graph_builder import create_graphs_from_context_data
from models.graph_model import CentralizedGATReliability


def train_gnn_model(model, train_loader, valid_loader, logger, args, normalization_params):
    logger.log("Training GNN model...")

    model = to_cuda(model, device=args.devices)[0]

    loss_function = get_loss_function(args)

    if args.use_gradnorm:
        if args.task_mode != "multi":
            logger.log(
                "Warning: GradNorm requires task_mode='multi'; disabling GradNorm for this run"
            )
            args.use_gradnorm = False
        else:
            logger.log("GradNorm: estimating the initial task losses L_0...")

            model.eval()

            task_losses_sum = torch.zeros(4).to(args.devices)
            num_batches = 0
            for batch_data in train_loader:
                if len(batch_data) == 6:
                    _, _, target_SR, target_RB, target_RH, target_RTT = batch_data
                    all_target_values = torch.stack(
                        [target_SR, target_RB, target_RH, target_RTT], dim=1
                    )
                else:
                    _, _, all_target_values = batch_data

                preds_batch = model.forward(
                    batch_data[0].to(args.devices), batch_data[1].to(args.devices), train_flag=True
                )
                all_target_values = all_target_values.to(args.devices)

                loss_SR = loss_function(preds_batch[:, 0], all_target_values[:, 0])
                loss_RB = loss_function(preds_batch[:, 1], all_target_values[:, 1])
                loss_RH = loss_function(preds_batch[:, 2], all_target_values[:, 2])
                loss_RTT = loss_function(preds_batch[:, 3], all_target_values[:, 3])

                batch_task_losses = torch.tensor(
                    [loss_SR, loss_RB, loss_RH, loss_RTT], device=args.devices
                )
                task_losses_sum += batch_task_losses
                num_batches += 1

            initial_task_losses = (task_losses_sum / num_batches).detach()
            logger.log(f"GradNorm: initial task losses L_0: {initial_task_losses.cpu().numpy()}")

            model.train()

    if args.use_gradnorm:
        loss_weights = torch.nn.Parameter(torch.ones(4, dtype=torch.float32, device=args.devices))

        weights_optimizer = torch.optim.Adam([loss_weights], lr=args.weights_lr)

    optimizer_embeds = get_optimizer(
        model.get_embeds_parameters(), lr=args.lr, decay=args.decay, args=args
    )

    optimizers = [optimizer_embeds]
    optimizer_names = ["embed"]

    if not args.ablation_no_gnn:
        optimizer_att = get_optimizer(
            model.get_attention_parameters(), lr=args.att_lr, decay=args.att_decay, args=args
        )
        optimizers.append(optimizer_att)
        optimizer_names.append("attention")

    mlp_params = model.get_mlp_parameters()
    if mlp_params:
        optimizer_mlp = get_optimizer(mlp_params, lr=args.lr, decay=args.decay, args=args)
        optimizers.append(optimizer_mlp)
        optimizer_names.append("mlp_interaction")

    early_stopping = None
    if args.enable_early_stop:
        early_stopping = EarlyStopping(
            args, patience=args.early_stop_patience, verbose=args.verbose
        )
        logger.log(f"Early stopping enabled, patience: {args.early_stop_patience}")
    else:
        logger.log("Early stopping disabled")

    best_model_state_dict = None
    best_valid_mae = float("inf")
    best_epoch = 0

    best_valid_metrics_so_far = {}

    total_training_time = 0.0

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7
        )
        for optimizer in optimizers
    ]

    for epoch in range(args.epochs):
        epoch_start_time = time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, train_batch in enumerate(train_loader):
            if len(train_batch) == 6:
                userIdx, servIdx, target_SR, target_RB, target_RH, target_RTT = train_batch

                all_target_values = torch.stack(
                    [target_SR, target_RB, target_RH, target_RTT], dim=1
                )
            elif len(train_batch) == 3:
                userIdx, servIdx, all_target_values = train_batch
            else:
                raise ValueError(f"Unexpected batch format: {len(train_batch)} elements")

            userIdx, servIdx, all_target_values = to_cuda(
                userIdx, servIdx, all_target_values, device=args.devices
            )

            for optimizer in optimizers:
                optimizer_zero_grad(optimizer)

            preds_batch = model.forward(userIdx, servIdx, train_flag=True)

            if args.use_gradnorm:
                loss_SR = loss_function(
                    preds_batch[:, 0].to(torch.float32), all_target_values[:, 0].to(torch.float32)
                )
                loss_RB = loss_function(
                    preds_batch[:, 1].to(torch.float32), all_target_values[:, 1].to(torch.float32)
                )
                loss_RH = loss_function(
                    preds_batch[:, 2].to(torch.float32), all_target_values[:, 2].to(torch.float32)
                )
                loss_RTT = loss_function(
                    preds_batch[:, 3].to(torch.float32), all_target_values[:, 3].to(torch.float32)
                )

                raw_task_losses = torch.stack([loss_SR, loss_RB, loss_RH, loss_RTT])

                weighted_loss = torch.sum(raw_task_losses * loss_weights)

                weighted_loss.backward(retain_graph=True)

                shared_params = model.get_last_shared_layer_parameters()

                if shared_params:
                    grad_norms = []
                    for i in range(4):
                        grad_i = torch.autograd.grad(
                            raw_task_losses[i] * loss_weights[i],
                            shared_params,
                            retain_graph=True,
                            create_graph=True,
                        )

                        grad_norm_i = torch.cat([g.view(-1) for g in grad_i]).norm()
                        grad_norms.append(grad_norm_i)

                    grad_norms = torch.stack(grad_norms)

                    G_avg = grad_norms.mean()

                    relative_losses = raw_task_losses / initial_task_losses

                    gradnorm_loss = torch.sum(
                        torch.abs(grad_norms - G_avg * (relative_losses**args.gradnorm_beta))
                    )

                    weights_optimizer.zero_grad()
                    gradnorm_loss.backward()

                    for optimizer in optimizers:
                        optimizer_step(optimizer)

                    weights_optimizer.step()

                    loss_weights.data = (loss_weights.data / loss_weights.data.sum()) * 4

                    total_loss = weighted_loss

                    if args.verbose and batch_idx % 100 == 0:
                        logger.log(
                            f"  Batch {batch_idx}/{len(train_loader)}, GradNorm Loss: {gradnorm_loss.item():.6f}, Weights: {loss_weights.data.cpu().numpy()}"
                        )
                else:
                    total_loss = weighted_loss

                    for optimizer in optimizers:
                        optimizer_step(optimizer)
            else:
                if args.task_mode == "single":
                    if preds_batch.ndim > 1 and preds_batch.shape[1] == 1:
                        preds_batch_sr = preds_batch.squeeze(-1)
                    else:
                        preds_batch_sr = preds_batch

                    total_loss = loss_function(
                        preds_batch_sr.to(torch.float32), all_target_values[:, 0].to(torch.float32)
                    )

                    if args.verbose and batch_idx % 100 == 0:
                        logger.log(
                            f"  Batch {batch_idx}/{len(train_loader)}, SR_Loss (Single Task): {total_loss.item():.6f}"
                        )

                elif args.task_mode == "multi":
                    loss_SR = loss_function(
                        preds_batch[:, 0].to(torch.float32),
                        all_target_values[:, 0].to(torch.float32),
                    )
                    loss_RB = loss_function(
                        preds_batch[:, 1].to(torch.float32),
                        all_target_values[:, 1].to(torch.float32),
                    )
                    loss_RH = loss_function(
                        preds_batch[:, 2].to(torch.float32),
                        all_target_values[:, 2].to(torch.float32),
                    )
                    loss_RTT = loss_function(
                        preds_batch[:, 3].to(torch.float32),
                        all_target_values[:, 3].to(torch.float32),
                    )

                    loss_weights_static = getattr(args, "loss_weights", [1.0, 1.0, 1.0, 1.0])

                    total_loss = (
                        loss_weights_static[0] * loss_SR
                        + loss_weights_static[1] * loss_RB
                        + loss_weights_static[2] * loss_RH
                        + loss_weights_static[3] * loss_RTT
                    )

                    if args.verbose and batch_idx % 100 == 0:
                        logger.log(
                            f"  Batch {batch_idx}/{len(train_loader)}, Loss_SR: {loss_SR.item():.4f}, Loss_RB: {loss_RB.item():.4f}, Loss_RH: {loss_RH.item():.4f}, Loss_RTT: {loss_RTT.item():.4f}, Total_Weighted_Loss: {total_loss.item():.6f}"
                        )
                else:
                    raise ValueError(f"Unknown task_mode: {args.task_mode}")

                total_loss.backward()

                if args.max_gradient > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

                for optimizer in optimizers:
                    optimizer_step(optimizer)

            if args.use_gradnorm and args.max_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

            epoch_loss += total_loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches

        model.eval()
        torch.set_grad_enabled(False)
        train_metrics = test_gnn_model(model, train_loader, logger, args, normalization_params)

        valid_metrics = test_gnn_model(model, valid_loader, logger, args, normalization_params)

        if args.task_mode == "single":
            primary_valid_metric = valid_metrics["SR"]["MAE"]
        else:
            primary_valid_metric = valid_metrics["SR"]["MAE"]

        torch.set_grad_enabled(True)

        epoch_training_time = time() - epoch_start_time
        total_training_time += epoch_training_time

        if (epoch + 1) % args.log_every_n_epochs == 0 or (epoch + 1) == args.epochs:
            log_line_1 = f"Epoch [{epoch + 1:3d}/{args.epochs}] - Time: {epoch_training_time:.4f}s | Total Loss: {avg_epoch_loss:.6f}"
            logger.log(log_line_1)

            if args.task_mode == "single":
                sr_train_mae = train_metrics.get("SR", {}).get("MAE", float("nan"))
                sr_train_rmse = train_metrics.get("SR", {}).get("RMSE", float("nan"))
                sr_valid_mae = valid_metrics.get("SR", {}).get("MAE", float("nan"))
                sr_valid_rmse = valid_metrics.get("SR", {}).get("RMSE", float("nan"))

                log_line_train = f"  Train: MAE={sr_train_mae:.4f} RMSE={sr_train_rmse:.4f}"
                logger.log(log_line_train)
                log_line_valid = f"  Valid: MAE={sr_valid_mae:.4f} RMSE={sr_valid_rmse:.4f}"
                logger.log(log_line_valid)

            elif args.task_mode == "multi":
                train_avg_mae = train_metrics.get("AVG_MAE", float("nan"))
                train_avg_rmse = train_metrics.get("AVG_RMSE", float("nan"))
                valid_avg_mae = valid_metrics.get("AVG_MAE", float("nan"))
                valid_avg_rmse = valid_metrics.get("AVG_RMSE", float("nan"))

                sr_train_mae = train_metrics.get("SR", {}).get("MAE", float("nan"))
                sr_train_rmse = train_metrics.get("SR", {}).get("RMSE", float("nan"))
                rb_train_mae = train_metrics.get("RB", {}).get("MAE", float("nan"))
                rb_train_rmse = train_metrics.get("RB", {}).get("RMSE", float("nan"))
                rh_train_mae = train_metrics.get("RH", {}).get("MAE", float("nan"))
                rh_train_rmse = train_metrics.get("RH", {}).get("RMSE", float("nan"))
                rtt_train_mae_norm = train_metrics.get("RTT", {}).get(
                    "MAE_normalized", float("nan")
                )
                rtt_train_rmse_norm = train_metrics.get("RTT", {}).get(
                    "RMSE_normalized", float("nan")
                )

                log_line_train = (
                    f"  Train: MAE={train_avg_mae:.4f} RMSE={train_avg_rmse:.4f} | "
                    f"SR(MAE:{sr_train_mae:.4f} RMSE:{sr_train_rmse:.4f}), "
                    f"RB(MAE:{rb_train_mae:.4f} RMSE:{rb_train_rmse:.4f}), "
                    f"RH(MAE:{rh_train_mae:.4f} RMSE:{rh_train_rmse:.4f}), "
                    f"RTT(MAE:{rtt_train_mae_norm:.4f} RMSE:{rtt_train_rmse_norm:.4f})"
                )
                logger.log(log_line_train)

                sr_valid_mae = valid_metrics.get("SR", {}).get("MAE", float("nan"))
                sr_valid_rmse = valid_metrics.get("SR", {}).get("RMSE", float("nan"))
                rb_valid_mae = valid_metrics.get("RB", {}).get("MAE", float("nan"))
                rb_valid_rmse = valid_metrics.get("RB", {}).get("RMSE", float("nan"))
                rh_valid_mae = valid_metrics.get("RH", {}).get("MAE", float("nan"))
                rh_valid_rmse = valid_metrics.get("RH", {}).get("RMSE", float("nan"))
                rtt_valid_mae_norm = valid_metrics.get("RTT", {}).get(
                    "MAE_normalized", float("nan")
                )
                rtt_valid_rmse_norm = valid_metrics.get("RTT", {}).get(
                    "RMSE_normalized", float("nan")
                )

                log_line_valid = (
                    f"  Valid: MAE={valid_avg_mae:.4f} RMSE={valid_avg_rmse:.4f} | "
                    f"SR(MAE:{sr_valid_mae:.4f} RMSE:{sr_valid_rmse:.4f}), "
                    f"RB(MAE:{rb_valid_mae:.4f} RMSE:{rb_valid_rmse:.4f}), "
                    f"RH(MAE:{rh_valid_mae:.4f} RMSE:{rh_valid_rmse:.4f}), "
                    f"RTT(MAE:{rtt_valid_mae_norm:.4f} RMSE:{rtt_valid_rmse_norm:.4f})"
                )
                logger.log(log_line_valid)

            if args.use_gradnorm and "loss_weights" in locals():
                weights_numpy = loss_weights.cpu().data.numpy()
                weights_str = f"[{weights_numpy[0]:.2f} {weights_numpy[1]:.2f} {weights_numpy[2]:.2f} {weights_numpy[3]:.2f}]"

                logger.log(f"  GradNorm Weights (SR, RB, RH, RTT): {weights_str}")

        if primary_valid_metric < best_valid_mae:
            best_valid_mae = primary_valid_metric
            best_epoch = epoch + 1
            best_model_state_dict = copy.deepcopy(model.state_dict())

            best_valid_metrics_so_far = valid_metrics.copy()

        if early_stopping is not None:
            early_stopping(primary_valid_metric, model)
            if early_stopping.early_stop:
                logger.log(f"Early stopping triggered; best epoch: {best_epoch}")
                break

    logger.log("\n" + "=" * 60)
    logger.log(
        f"Training completed! Best validation SR_MAE: {best_valid_mae:.6f} (Epoch {best_epoch})"
    )
    logger.log(f"Total training time: {total_training_time:.2f}s")

    return best_model_state_dict, total_training_time, best_valid_metrics_so_far


def test_gnn_model(model, data_loader, logger, args, normalization_params):
    model.eval()
    torch.set_grad_enabled(False)

    if hasattr(model, "prepare_test_model"):
        model.prepare_test_model()

    preds_all = []
    reals_all = []

    for batch_data in data_loader:
        if len(batch_data) == 6:
            userIdx, servIdx, target_SR, target_RB, target_RH, target_RTT = batch_data

            all_target_values = torch.stack([target_SR, target_RB, target_RH, target_RTT], dim=1)
        elif len(batch_data) == 3:
            userIdx, servIdx, all_target_values = batch_data
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")

        userIdx, servIdx, all_target_values = to_cuda(
            userIdx, servIdx, all_target_values, device=args.devices
        )

        pred = model.forward(userIdx, servIdx, train_flag=False)

        preds_all.append(pred.cpu().numpy())
        reals_all.append(all_target_values.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    reals_all = np.concatenate(reals_all, axis=0)

    metrics_to_return = {}
    target_names = ["SR", "RB", "RH", "RTT"]

    if args.task_mode == "single":
        current_preds_single_task = (
            preds_all.squeeze() if preds_all.ndim > 1 and preds_all.shape[1] == 1 else preds_all
        )
        current_reals_single_task = reals_all[:, 0]

        sr_scale_factor = normalization_params.get("scale_factors", {}).get("SR", 1.0)

        preds_sr_denorm = current_preds_single_task * sr_scale_factor
        reals_sr_denorm = current_reals_single_task * sr_scale_factor

        mae = np.mean(np.abs(reals_sr_denorm - preds_sr_denorm))
        rmse = np.sqrt(np.mean((reals_sr_denorm - preds_sr_denorm) ** 2))

        metrics_to_return["SR"] = {"MAE": mae, "RMSE": rmse}

    elif args.task_mode == "multi":
        calculated_metrics_for_avg = {"MAE": [], "RMSE": []}

        for i, metric_name in enumerate(target_names):
            current_preds_norm = preds_all[:, i]
            current_reals_norm = reals_all[:, i]

            metrics_to_return[metric_name] = {}

            if metric_name == "RTT":
                mae_rtt_norm = np.mean(np.abs(current_reals_norm - current_preds_norm))
                rmse_rtt_norm = np.sqrt(np.mean((current_reals_norm - current_preds_norm) ** 2))
                metrics_to_return[metric_name]["MAE_normalized"] = mae_rtt_norm
                metrics_to_return[metric_name]["RMSE_normalized"] = rmse_rtt_norm

                calculated_metrics_for_avg["MAE"].append(mae_rtt_norm)
                calculated_metrics_for_avg["RMSE"].append(rmse_rtt_norm)

            current_scale_factor = normalization_params.get("scale_factors", {}).get(metric_name)

            current_preds_denorm = current_preds_norm
            current_reals_denorm = current_reals_norm

            if current_scale_factor and current_scale_factor != 0:
                if current_scale_factor != 1.0:
                    current_preds_denorm = current_preds_norm * current_scale_factor
                    current_reals_denorm = current_reals_norm * current_scale_factor

            if metric_name == "RTT":
                current_preds_denorm = np.expm1(current_preds_denorm)
                current_reals_denorm = np.expm1(current_reals_denorm)

            mae = np.mean(np.abs(current_reals_denorm - current_preds_denorm))
            rmse = np.sqrt(np.mean((current_reals_denorm - current_preds_denorm) ** 2))
            metrics_to_return[metric_name]["MAE"] = mae
            metrics_to_return[metric_name]["RMSE"] = rmse

            if metric_name != "RTT":
                calculated_metrics_for_avg["MAE"].append(mae)
                calculated_metrics_for_avg["RMSE"].append(rmse)

        if len(calculated_metrics_for_avg["MAE"]) == 4:
            metrics_to_return["AVG_MAE"] = np.mean(calculated_metrics_for_avg["MAE"])
            metrics_to_return["AVG_RMSE"] = np.mean(calculated_metrics_for_avg["RMSE"])

    else:
        raise ValueError(f"Unknown task_mode: {args.task_mode}")

    torch.set_grad_enabled(True)
    return metrics_to_return


def run_single_experiment(args, logger):
    logger.log("Starting single experiment...")

    ablation_flags = []
    if args.ablation_no_gnn:
        ablation_flags.append("No-GNN")
    if args.ablation_no_context:
        ablation_flags.append("No-Context")
    if args.ablation_use_gcn:
        ablation_flags.append("GCN")
    if args.ablation_single_layer:
        ablation_flags.append("Single-Layer")

    if ablation_flags:
        logger.log(f"Ablation study: {' + '.join(ablation_flags)}")
    else:
        logger.log("Full model: GAT with context")

    set_seed(args.random_seed)

    logger.log("Loading data...")
    try:
        train_loader, valid_loader, test_loader, norm_params = get_dataloaders_for_reliability(
            args=args,
            param_suffix=args.metric_file_suffix,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
        )
        normalization_params = norm_params

        logger.log(
            f"Data loaded! Batches - Train: {len(train_loader)}, Valid: {len(valid_loader)}, Test: {len(test_loader)}"
        )
    except Exception as e:
        logger.log(f"Data loading failed: {e}")
        raise

    import pandas as pd

    try:
        client_ctx_df = pd.read_csv(os.path.join(args.dataset_path, args.client_ctx_file))
        peer_ctx_df = pd.read_csv(os.path.join(args.dataset_path, args.peer_ctx_file))

    except Exception as e:
        logger.log(f"Context data loading failed: {e}")
        raise

    try:
        user_graph, serv_graph, user_lookup, serv_lookup = create_graphs_from_context_data(
            client_ctx_df=client_ctx_df,
            peer_ctx_df=peer_ctx_df,
            client_feature_cols=args.client_feature_cols,
            peer_feature_cols=args.peer_feature_cols,
            args=args,
        )

        logger.log(
            f"Graphs built - User: {user_graph.number_of_nodes()} nodes, Service: {serv_graph.number_of_nodes()} nodes"
        )
    except Exception as e:
        logger.log(f"Graph building failed: {e}")
        raise

    try:
        model = CentralizedGATReliability(
            args=args,
            user_graph=user_graph,
            serv_graph=serv_graph,
            num_users=len(client_ctx_df),
            num_services=len(peer_ctx_df),
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.log(f"Model created! Total parameters: {total_params:,}")

    except Exception as e:
        logger.log(f"Model creation failed: {e}")
        raise

    logger.log("\n" + "=" * 60)
    logger.log("Training GNN model...")
    logger.log("=" * 60)
    try:
        best_model_state_dict, training_time, best_valid_metrics = train_gnn_model(
            model, train_loader, valid_loader, logger, args, normalization_params
        )
    except Exception as e:
        logger.log(f"Training failed: {e}")
        raise

    logger.log("Testing GNN model...")
    try:
        model.load_state_dict(best_model_state_dict)

        test_metrics = test_gnn_model(model, test_loader, logger, args, normalization_params)

        if args.task_mode == "single":
            sr_metrics = test_metrics.get("SR", {})
            logger.log(
                f"  [Test Results] MAE: {sr_metrics.get('MAE', float('nan')):.6f}, RMSE: {sr_metrics.get('RMSE', float('nan')):.6f}"
            )

        elif args.task_mode == "multi":
            avg_mae = test_metrics.get("AVG_MAE", float("nan"))
            avg_rmse = test_metrics.get("AVG_RMSE", float("nan"))

            sr_mae = test_metrics.get("SR", {}).get("MAE", float("nan"))
            sr_rmse = test_metrics.get("SR", {}).get("RMSE", float("nan"))
            rb_mae = test_metrics.get("RB", {}).get("MAE", float("nan"))
            rb_rmse = test_metrics.get("RB", {}).get("RMSE", float("nan"))
            rh_mae = test_metrics.get("RH", {}).get("MAE", float("nan"))
            rh_rmse = test_metrics.get("RH", {}).get("RMSE", float("nan"))
            rtt_mae_norm = test_metrics.get("RTT", {}).get("MAE_normalized", float("nan"))
            rtt_rmse_norm = test_metrics.get("RTT", {}).get("RMSE_normalized", float("nan"))

            log_line_test = (
                f"  [Test Results] "
                f"AVG_MAE: {avg_mae:.4f}, AVG_RMSE: {avg_rmse:.4f} | "
                f"SR(MAE:{sr_mae:.4f} RMSE:{sr_rmse:.4f}), "
                f"RB(MAE:{rb_mae:.4f} RMSE:{rb_rmse:.4f}), "
                f"RH(MAE:{rh_mae:.4f} RMSE:{rh_rmse:.4f}), "
                f"RTT(MAE:{rtt_mae_norm:.4f} RMSE:{rtt_rmse_norm:.4f})"
            )
            logger.log(log_line_test)

    except Exception as e:
        logger.log(f"Testing failed: {e}")
        raise

    logger.log("Experiment completed successfully!")

    return test_metrics, training_time, best_valid_metrics


def main_experiment_loop(args):
    logger = Logger(args)

    logger.log("Experiment parameter configuration:")
    logger.log(str(args))
    logger.log("=" * 80)

    results = collections.defaultdict(list)
    training_times = []

    for round_idx in range(args.rounds):
        logger.log(f"\n[Round {round_idx + 1}/{args.rounds}] Starting experiment...")

        current_seed = args.random_seed + round_idx
        args.random_seed = current_seed

        try:
            test_metrics, training_time, best_valid_metrics = run_single_experiment(args, logger)

            for metric_name, values_or_dict in test_metrics.items():
                if isinstance(values_or_dict, dict):
                    if "MAE" in values_or_dict:
                        result_append(
                            results, f"{metric_name}_MAE", values_or_dict.get("MAE", float("nan"))
                        )
                    if "RMSE" in values_or_dict:
                        result_append(
                            results, f"{metric_name}_RMSE", values_or_dict.get("RMSE", float("nan"))
                        )
                    if "MAE_normalized" in values_or_dict:
                        result_append(
                            results,
                            f"{metric_name}_MAE_normalized",
                            values_or_dict.get("MAE_normalized", float("nan")),
                        )
                    if "RMSE_normalized" in values_or_dict:
                        result_append(
                            results,
                            f"{metric_name}_RMSE_normalized",
                            values_or_dict.get("RMSE_normalized", float("nan")),
                        )
                else:
                    result_append(results, metric_name, values_or_dict)

            training_times.append(training_time)

            if args.task_mode == "single":
                sr_mae = test_metrics["SR"]["MAE"]
                sr_rmse = test_metrics["SR"]["RMSE"]
                logger.log(
                    f"[Round {round_idx + 1}/{args.rounds}] Completed: MAE={sr_mae:.6f}, RMSE={sr_rmse:.6f}, Time={training_time:.2f}s"
                )
            else:
                avg_mae = test_metrics.get("AVG_MAE", float("nan"))
                avg_rmse = test_metrics.get("AVG_RMSE", float("nan"))
                logger.log(
                    f"[Round {round_idx + 1}/{args.rounds}] Completed: MAE={avg_mae:.6f}, RMSE={avg_rmse:.6f}, Time={training_time:.2f}s"
                )

        except Exception as e:
            logger.log(f"[Round {round_idx + 1}/{args.rounds}] Failed: {e}")
            import traceback

            logger.log(traceback.format_exc())
            continue

    has_results = (
        len(results.get("SR_MAE", [])) > 0
        if args.task_mode == "single"
        else len(results.get("AVG_MAE", [])) > 0 or len(results.get("SR_MAE", [])) > 0
    )

    if len(results.get("SR_MAE", [])) > 0:
        logger.log("\n" + "=" * 80)
        logger.log(
            f"  [Final Results Summary (Avg over {len(results.get('SR_MAE', []))}/{args.rounds} successful rounds)]"
        )
        logger.log("=" * 80)

        if args.task_mode == "single":
            avg_mae_sr_single = np.mean(results.get("SR_MAE", [float("nan")]))
            avg_rmse_sr_single = np.mean(results.get("SR_RMSE", [float("nan")]))
            logger.log(f"MAE: {avg_mae_sr_single:.6f}, RMSE: {avg_rmse_sr_single:.6f}")
            logger.log("-" * 50)

        elif args.task_mode == "multi":
            avg_overall_mae = np.mean(results.get("AVG_MAE", [float("nan")]))
            avg_overall_rmse = np.mean(results.get("AVG_RMSE", [float("nan")]))

            avg_sr_mae = np.mean(results.get("SR_MAE", [float("nan")]))
            avg_sr_rmse = np.mean(results.get("SR_RMSE", [float("nan")]))

            avg_rb_mae = np.mean(results.get("RB_MAE", [float("nan")]))
            avg_rb_rmse = np.mean(results.get("RB_RMSE", [float("nan")]))

            avg_rh_mae = np.mean(results.get("RH_MAE", [float("nan")]))
            avg_rh_rmse = np.mean(results.get("RH_RMSE", [float("nan")]))

            avg_rtt_mae_norm = np.mean(results.get("RTT_MAE_normalized", [float("nan")]))
            avg_rtt_rmse_norm = np.mean(results.get("RTT_RMSE_normalized", [float("nan")]))

            metrics_line = (
                f"MAE: {avg_overall_mae:.6f} RMSE: {avg_overall_rmse:.6f} | "
                f"SR(MAE:{avg_sr_mae:.4f} RMSE:{avg_sr_rmse:.4f}), "
                f"RB(MAE:{avg_rb_mae:.4f} RMSE:{avg_rb_rmse:.4f}), "
                f"RH(MAE:{avg_rh_mae:.4f} RMSE:{avg_rh_rmse:.4f}), "
                f"RTT(MAE:{avg_rtt_mae_norm:.4f} RMSE:{avg_rtt_rmse_norm:.4f})"
            )
            logger.log(metrics_line)
            logger.log("-" * (len(metrics_line) if len(metrics_line) < 80 else 80))

        if training_times:
            mean_time = np.mean(training_times)
            logger.log(f"\nAverage training time: {mean_time:.2f} seconds")

        successful_rounds = len(results.get("SR_MAE", []))
        logger.log(f"Successfully completed {successful_rounds}/{args.rounds} rounds")
    else:
        logger.log("No valid results were produced because all runs failed")

    logger.log("=" * 80)
    logger.log("All experiments completed")


if __name__ == "__main__":
    args = parse_args()

    main_experiment_loop(args)
