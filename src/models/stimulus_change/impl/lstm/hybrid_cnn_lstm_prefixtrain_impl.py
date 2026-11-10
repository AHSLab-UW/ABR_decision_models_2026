"""
Hybrid CNN-LSTM Prefix-Train Implementation
===========================================
Single-head model: predicts next action class directly (all classes in one head).

This variant keeps the same architecture and data pipeline as hybrid_cnn_lstm_impl,
but changes the training regime to match prefix-style evaluation:
  - train on prefixes
  - validate on prefixes
  - test on prefixes
  - score / supervise only the last timestep of each prefix

Usage:
    python -m src.models.stimulus_change.impl.lstm.hybrid_cnn_lstm_prefixtrain_impl
    python -m src.models.stimulus_change.impl.lstm.hybrid_cnn_lstm_prefixtrain_impl --train-fixed
    python -m src.models.stimulus_change.impl.lstm.hybrid_cnn_lstm_prefixtrain_impl --eval
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

from src.models.stimulus_change.impl.lstm.hybrid_cnn_lstm_impl import (
    DEVICE,
    ROOT_DIR,
    HybridCNNLSTMModel,
    class_to_db,
    file_exists,
    get_sampling_probability,
    is_higher_than_saved_acc,
    labels,
    mapping,
    model_configs,
    num_classes,
    ordinal_distance_loss,
    plot_cm,
    print_model_architecture,
    reverse_mapping,
    split_for_testing_waveform,
    test_dataset,
    test_split_prev_preds,
    train_dataset,
    val_dataset,
)


def compute_prefix_accuracy(model, prefixes):
    """Evaluate prefixes and score only the final timestep."""
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0
    with torch.no_grad():
        for wave, meta, label in prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)

            pred, _ = model.forward_test(wave, meta)
            correct += int(pred[-1].item() == label[-1].item())
            total += 1
    model.train()
    return correct / total * 100 if total > 0 else 0


def training_loop(epochs, model, lr, train_prefixes, val_prefixes, label_map, sampling_strategy, class_to_db_tensor):
    """Train on prefixes and supervise only the last timestep."""
    device = next(model.parameters()).device

    last_labels = [y[-1].item() for _, _, y in train_prefixes]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(list(label_map.values())),
        y=np.array(last_labels),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    class_to_db_tensor = class_to_db_tensor.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    steps_per_epoch = len(train_prefixes)
    warmup_epochs = 20
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    print(f"  Train prefixes: {len(train_prefixes)} (1 forward pass/prefix, supervise last timestep)")
    print(f"  Val prefixes: {len(val_prefixes)} (1 forward pass/prefix, score last timestep)")

    patience = 25
    best_val_acc = 0.0
    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve_loss = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_ord = 0.0
        total_decorr = 0.0
        n_batches = 0
        model.train()

        for wave, meta, lbl in train_prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            lbl = lbl.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            optimizer.zero_grad()
            sampling_prob = get_sampling_probability(sampling_strategy, epoch, epochs)
            output_seq, labels_seq, decorr_loss = model.forward_train(wave, meta, lbl, sampling_prob)

            output_last = output_seq[-1:].contiguous()
            label_last = labels_seq[-1:].contiguous()

            ce_loss = criterion(output_last, label_last)
            ord_loss = ordinal_distance_loss(output_last, label_last, class_to_db_tensor, scale=0.03)
            loss = ce_loss + ord_loss + 0.05 * decorr_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_ord += ord_loss.item()
            total_decorr += decorr_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce = total_ce / max(n_batches, 1)
        avg_ord = total_ord / max(n_batches, 1)
        avg_decorr = total_decorr / max(n_batches, 1)
        val_acc = compute_prefix_accuracy(model, val_prefixes)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}, Loss: {avg_loss:.4f} "
            f"(CE: {avg_ce:.4f}, Ord: {avg_ord:.4f}, Decorr: {avg_decorr:.4f}), "
            f"Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best model saved! (val acc: {val_acc:.2f}%)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve_loss = 0
        else:
            epochs_no_improve_loss += 1

        if epochs_no_improve_loss >= patience:
            print(f"Early stopping at epoch {epoch+1} (loss stopped improving)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val acc: {best_val_acc:.2f}%)")


def testing_loop(model, prefixes, reverse_map):
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    pm10db_correct, pm5db_correct, pm15db_correct, quasi_total = 0, 0, 0, 0
    top3_correct, top5_correct = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for wave, meta, label in prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            pred, last_logits = model.forward_test(wave, meta)
            pred_last = pred[-1].item()
            label_last = label[-1].item()

            correct += int(pred_last == label_last)
            total += 1

            correct_mapped = reverse_map[label_last]
            pred_mapped = reverse_map[pred_last]

            top3 = torch.topk(last_logits, k=3).indices.tolist()
            top5 = torch.topk(last_logits, k=5).indices.tolist()
            top3_correct += int(label_last in top3)
            top5_correct += int(label_last in top5)

            if (
                "Stop" not in correct_mapped
                and "Stop" not in pred_mapped
                and ">" not in correct_mapped
                and ">" not in pred_mapped
                and "<" not in correct_mapped
                and "<" not in pred_mapped
            ):
                quasi_total += 1
                value = correct_mapped.split(" ")[0]
                value_pred = pred_mapped.split(" ")[0]
                if "Repeat" in value:
                    value = "0"
                if "Repeat" in value_pred:
                    value_pred = "0"
                if abs(int(value) - int(value_pred)) <= 5:
                    pm5db_correct += 1
                if abs(int(value) - int(value_pred)) <= 10:
                    pm10db_correct += 1
                if abs(int(value) - int(value_pred)) <= 15:
                    pm15db_correct += 1

            predictions.append(pred_last)
            true_labels.append(label_last)

    accuracy = correct / total * 100
    top3_acc = top3_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"\nFraction: {correct}/{total}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    if quasi_total > 0:
        print(f"Within 10 dB: {pm10db_correct}/{quasi_total} = {100*(pm10db_correct/quasi_total):.2f}%")
        print(f"Within 5 dB: {pm5db_correct}/{quasi_total} = {100*(pm5db_correct/quasi_total):.2f}%")
        print(f"Within 15 dB: {pm15db_correct}/{quasi_total} = {100*(pm15db_correct/quasi_total):.2f}%")

    results = {
        "accuracy": accuracy,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "pm10db_correct": 100 * (pm10db_correct / quasi_total) if quasi_total > 0 else 0,
        "pm5db_correct": 100 * (pm5db_correct / quasi_total) if quasi_total > 0 else 0,
        "pm15db_correct": 100 * (pm15db_correct / quasi_total) if quasi_total > 0 else 0,
    }

    cm = confusion_matrix(y_true=true_labels, y_pred=predictions)
    return accuracy, cm, results


def run_model(model, epochs, lr, sampling_strategy, train_prefixes, val_prefixes, test_prefixes, label_map, reverse_map, class_to_db_tensor):
    training_loop(
        epochs=epochs,
        model=model,
        lr=lr,
        train_prefixes=train_prefixes,
        val_prefixes=val_prefixes,
        label_map=label_map,
        sampling_strategy=sampling_strategy,
        class_to_db_tensor=class_to_db_tensor,
    )
    return testing_loop(model, test_prefixes, reverse_map)


RESULTS_TAG = "hybrid_cnn_lstm_prefixtrain"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model (no training)")
    parser.add_argument("--train-fixed", action="store_true", help="Train with fixed config (no grid search)")
    parser.add_argument("--config", type=str, default=f"{ROOT_DIR}/exported_features/config/{RESULTS_TAG}_config.json")
    parser.add_argument("--weights", type=str, default=f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
    args = parser.parse_args()

    print("Building prefix datasets...")
    train_prefixes = split_for_testing_waveform(train_dataset)
    val_prefixes = split_for_testing_waveform(val_dataset)
    test_prefixes = test_split_prev_preds
    print(f"  Train prefixes: {len(train_prefixes)}")
    print(f"  Val prefixes: {len(val_prefixes)}")
    print(f"  Test prefixes: {len(test_prefixes)}")

    if args.eval:
        print(f"\n{'='*60}")
        print("EVAL MODE: Loading saved model and evaluating...")
        print("=" * 60)

        with open(args.config) as f:
            config = json.load(f)
        config["output_size"] = num_classes
        config["meta_dim"] = 12

        model = HybridCNNLSTMModel(config).to(DEVICE)
        model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=DEVICE))
        print(f"Loaded weights from: {args.weights}")
        print_model_architecture(model, config)

        accuracy, cm, results = testing_loop(model, test_prefixes, reverse_mapping)
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        plot_cm(cm, f"Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%", "Predicted Labels", "True Labels", labels)

    elif args.train_fixed:
        print(f"\n{'='*60}")
        print(f"TRAIN-FIXED MODE: Training {RESULTS_TAG} with fixed config...")
        print("=" * 60)

        with open(args.config) as f:
            config = json.load(f)
        config["output_size"] = num_classes
        config["meta_dim"] = 12

        model = HybridCNNLSTMModel(config).to(DEVICE)
        print_model_architecture(model, config)

        accuracy, cm, results = run_model(
            model,
            epochs=150,
            lr=0.0005,
            sampling_strategy="linear",
            train_prefixes=train_prefixes,
            val_prefixes=val_prefixes,
            test_prefixes=test_prefixes,
            label_map=mapping,
            reverse_map=reverse_mapping,
            class_to_db_tensor=class_to_db,
        )

        print(f"\nTest Accuracy: {accuracy:.2f}%")

        results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_results.json"
        if is_higher_than_saved_acc(accuracy, results_path):
            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)
            torch.save(model.state_dict(), f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print("Saved new best model!")

        plot_cm(cm, f"Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%", "Predicted Labels", "True Labels", labels)

    else:
        print(f"\nExecuting {RESULTS_TAG} grid search...")

        best_acc = 0
        plt_best = None
        i = 1
        total = len(model_configs)

        for config in model_configs:
            print(f"\n{'='*60}")
            print(f"{i} / {total}: Prefix-train grid run")
            print("=" * 60)

            model = HybridCNNLSTMModel(config).to(DEVICE)
            print_model_architecture(model, config)

            accuracy, cm, results = run_model(
                model,
                epochs=200,
                lr=0.005,
                sampling_strategy="linear",
                train_prefixes=train_prefixes,
                val_prefixes=val_prefixes,
                test_prefixes=test_prefixes,
                label_map=mapping,
                reverse_map=reverse_mapping,
                class_to_db_tensor=class_to_db,
            )

            results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_results.json"
            if accuracy > best_acc and is_higher_than_saved_acc(accuracy, results_path):
                print(f"\n*** Found better accuracy: {accuracy:.2f}%! ***")

                os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
                os.makedirs(f"{ROOT_DIR}/exported_features/config", exist_ok=True)
                os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)

                torch.save(model.state_dict(), f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
                with open(f"{ROOT_DIR}/exported_features/config/{RESULTS_TAG}_config.json", "w") as f:
                    json.dump(config, f, indent=4)
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

                best_acc = accuracy
                plt_best = cm
                plot_cm(
                    cm,
                    f"Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%",
                    "Predicted Labels",
                    "True Labels",
                    labels,
                    save_path=f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_cm.png",
                    should_save=True,
                )

            i += 1

        print(f"\n{'='*60}")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print("=" * 60)

        if plt_best is not None:
            plot_cm(
                plt_best,
                f"Confusion Matrix ({RESULTS_TAG}) - {best_acc:.2f}%",
                "Predicted Labels",
                "True Labels",
                labels,
                save_path=f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_cm.png",
                should_save=False,
            )
