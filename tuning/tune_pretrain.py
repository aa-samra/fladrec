import optuna
import subprocess
import sys
import json
from pathlib import Path
import yaml
import logging
import os
import shutil

# Configure logging for the tuning script
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def objective(trial, domain):
    """
    Objective function for Optuna to maximize.
    """
    # Define hyperparameter search space
    hp = {
        'embedding_dim': trial.suggest_categorical('embedding_dim', [64, 128, 256]),
        'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4]),
        'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5)
    }

    loss_type = 'ce'
    
    # Load base domain config
    base_config_path = Path(f'config/domain/{domain}.yaml')
    if not base_config_path.exists():
        logging.error(f"Base domain config not found at {base_config_path}")
        return -1
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Create new domain config for the trial
    trial_domain_name = f"{domain}_{trial.number:02d}"
    trial_config = {
        'name': trial_domain_name,
        'path': base_config['path'],
        'max_seq_len': base_config.get('max_seq_len', 50),
        'hp': hp
    }
    
    trial_config_path = Path(f'config/domain/{trial_domain_name}.yaml')
    with open(trial_config_path, 'w') as f:
        yaml.dump(trial_config, f, sort_keys=False)

    logging.info(f"Saved trial {trial.number} config to {trial_config_path}")

    # Construct command
    command = [
        sys.executable,
        'scripts/pretrain.py',
        f'domain={trial_domain_name}',
        f'loss_type={loss_type}',
        f'phase=train'
    ]

    logging.info(f"Trial {trial.number}: Running command: {' '.join(command)}")

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        # The last line of stdout should be the JSON metrics
        output_lines = process.stdout.strip().split('\n')
        if not output_lines:
            logging.error(f"Trial {trial.number}: No output from pretrain.py.")
            return -1

        json_output = ''
        # Find the last line that is a valid json
        for line in reversed(output_lines):
            if line.strip().startswith('{') and line.strip().endswith('}'):
                json_output = line
                break
        
        if not json_output:
            logging.error(f"Trial {trial.number}: No JSON output from pretrain.py. Full output:\n{process.stdout}")
            return -1

        metrics = json.loads(json_output)
        
        ndcg_10 = metrics.get('ndcg', {}).get('10', -1)
        logging.info(f"Trial {trial.number}: ndcg@10 = {ndcg_10}")

        if ndcg_10 == -1:
            logging.warning(f"Trial {trial.number}: 'ndcg' at 10 not found in metrics: {metrics}")

        return ndcg_10

    except subprocess.CalledProcessError as e:
        logging.error(f"Trial {trial.number}: pretrain.py failed with exit code {e.returncode}.")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return -1
    except json.JSONDecodeError as e:
        logging.error(f"Trial {trial.number}: Failed to decode JSON from pretrain.py output.")
        logging.error(f"Received output: {json_output}")
        return -1
    except Exception as e:
        logging.error(f"An unexpected error occurred during trial {trial.number}: {e}")
        return -1
    finally:
        # Clean up the generated config file
        if trial_config_path.exists():
            os.remove(trial_config_path)
            logging.info(f"Cleaned up trial config: {trial_config_path}")


def checkpoint_callback(study, trial):
    """After each trial: keep only the best checkpoint, renamed without suffix."""
    checkpoints_dir = Path('checkpoints')  # adjust if your checkpoint dir differs

    for item in checkpoints_dir.glob(f"*_{trial.number:02d}*"):
        if study.best_trial.number == trial.number:
            # This trial is the new best — rename to remove the numeric suffix
            best_name = item.name.replace(f"_{trial.number:02d}", "")
            best_path = item.parent / best_name
            # Remove any previously promoted best checkpoint
            if best_path.exists():
                if best_path.is_dir():
                    shutil.rmtree(best_path)
                else:
                    best_path.unlink()
            item.rename(best_path)
            logging.info(f"Promoted checkpoint: {item.name} → {best_name}")
        else:
            # Not the best — delete it
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            logging.info(f"Deleted non-best checkpoint: {item.name}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tuning/tune_pretrain.py <domain_name> [n_trials]")
        sys.exit(1)

    domain_name = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    study_name = f'{domain_name}-pretrain-tuning'
    
    # Create directory for tuning database if it doesn't exist
    tuning_dir = Path('tuning')
    tuning_dir.mkdir(exist_ok=True)
    storage_path = f"sqlite:///{tuning_dir / 'optuna.db'}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction='maximize',
        load_if_exists=True
    )

    logging.info(f"Starting Optuna study '{study_name}' for domain '{domain_name}' with {n_trials} trials.")
    study.optimize(lambda trial: objective(trial, domain_name), n_trials=n_trials, callbacks=[checkpoint_callback])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print("      {}: {}".format(key, value))

    # The config for the best trial is already saved as config/domain/<domain>_<best_trial_number>.yaml
    # during the run, but it's cleaned up afterwards. Let's save the best one again.
    best_trial_domain_name = f"{domain_name}_{study.best_trial.number:02d}"
    best_config_path = Path(f'config/domain/{domain_name}.yaml')
    
    # Recreate the best config
    best_hp = study.best_trial.params.copy()
    loss_type = best_hp.pop('loss_type', 'ce')  # loss_type is not part of hp
    
    # Load base domain config
    base_config_path = Path(f'config/domain/{domain_name}.yaml')
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    best_config_content = {
        'name': domain_name,
        'path': base_config['path'],
        'max_seq_len': base_config.get('max_seq_len', 50),
        'hp': best_hp
    }

    with open(best_config_path, 'w') as f:
        yaml.dump(best_config_content, f, sort_keys=False)
    
    logging.info(f"Saved best hyperparameters to {best_config_path}")
    logging.info(f"Note: The best loss_type was '{loss_type}'. This is not saved in the yaml but should be used when running pretraining.")