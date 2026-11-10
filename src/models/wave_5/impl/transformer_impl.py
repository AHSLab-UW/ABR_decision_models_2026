import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.common.model.wave_5_model_util import *
from src.common.plot_util import plot_cm
from src.models.wave_5.transformer.transformer_model import Wave5TransformerModel
from src.models.wave_5.transformer.configs.transformer_configurations import *

print(f"Initializing data pipeline...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/wave_5_action_table.csv")
input_data = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/input.csv")

# keep stop column for sequence splitting (unlike DNN which drops it)
merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_df, input_data)

mapping = {True: 1, False: 0}
reverse_mapping = {1: True, 0: False}
labels_categorical = np.array(label_ids.map(mapping))
labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

scale_dfs(merged_dfs)
print(f"Data has been scaled, now splitting into 80-10-10 split")

train_dfs, test_dfs, val_dfs = test_train_holdout_split_list(merged_dfs)

print(f"Splitting train, val, and test sets into labels and data...")

train_split = prepare_sequences_wave5(train_dfs, mapping)
test_split = prepare_sequences_wave5(test_dfs, mapping)
val_split = prepare_sequences_wave5(val_dfs, mapping)

print(f"Executing model tuning via grid search...")

def evaluate_grid_search(model_configs, epochs, learning_rates):
    best_acc = 0
    plt_best = None
    i = 1
    total = len(model_configs) * len(epochs) * len(learning_rates)

    random.shuffle(model_configs)
    for config in model_configs:
        for lr in learning_rates:
            for ep in epochs:
                print(f"{i} / {total}: Current Configuration: Epoch: {ep}, Learning Rate: {lr}")
                i += 1
                model = Wave5TransformerModel(config)
                print_model_architecture(model, config)
                results_logger, sequence_logger = run_model(
                    model,
                    epochs=ep,
                    lr=lr,
                    train_set=train_split,
                    test_set=test_split,
                    label_set=labels_categorical,
                    holdout_validation=config['holdout_validation'],
                    sequential=True,
                    val_set=val_split
                )
                if config['holdout_validation']:
                    torch.save(model.state_dict(),
                               f'{ROOT_DIR}/exported_features/wave_5/weights/transformer_weights.pth')
                    for j in range(len(val_dfs)):
                        val_dfs[j].to_csv(
                            f"{ROOT_DIR}/data/cross_validation_sets/wave_5/transformer/test_df_{j}.csv", index=False)
                    continue
                accuracy = results_logger.summarize_results()
                save_path = f"{ROOT_DIR}/exported_features/wave_5/results/train/transformer/transformer_results.json"
                if accuracy > best_acc and is_higher_than_saved_acc(accuracy, save_path):
                    print(f"Found better accuracy: {accuracy:.2f}%!")
                    sequence_logger.summarize_results()
                    results = results_logger.get_results()
                    cm = results_logger.get_cm()
                    torch.save(model.state_dict(),
                               f'{ROOT_DIR}/exported_features/wave_5/weights/transformer_weights.pth')
                    with open(f"{ROOT_DIR}/exported_features/wave_5/config/transformer_config.json", "w") as f:
                        json.dump(config, f, indent=4)
                    with open(save_path, "w") as f:
                        json.dump(results, f, indent=4)
                    with open(f"{ROOT_DIR}/exported_features/wave_5/results/train/transformer/sequence_results.json",
                              "w") as f:
                        json.dump(sequence_logger.get_binned_results_dict(), f, indent=4)
                    best_acc = accuracy
                    plt_best = cm
                    plot_cm(cm, f'Confusion Matrix (Wave 5 Transformer) - {accuracy:.2f}', 'Predicted', 'True', labels,
                            save_path=f"{ROOT_DIR}/exported_features/wave_5/results/train/transformer/transformer_cm.png",
                            should_save=True, holdout_validation_cm=False)
                    sequence_logger.show_plot(
                        f"{ROOT_DIR}/exported_features/wave_5/results/train/transformer/sequence_accuracy.png")
                    print("Found better accuracy, running cross-validation training to update weights")
                    config['holdout_validation'] = True
                    evaluate_grid_search([config], epochs, learning_rates)

    if plt_best is not None:
        print(best_acc)
        plot_cm(plt_best, f'Confusion Matrix (Wave 5 Transformer) - {best_acc:.2f}', 'Predicted Labels',
                'True Labels', labels,
                save_path=f"{ROOT_DIR}/exported_features/wave_5/results/train/transformer/transformer_cm.png",
                should_save=False, holdout_validation_cm=False)

evaluate_grid_search(model_configs, epochs, learning_rates)
