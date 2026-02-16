import optuna
import subprocess
import sys
import json
from pathlib import Path
import yaml
import logging
import os

# Configure logging for the tuning script
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def objective(trial, transfer_name):
    """
    Objective function for Optuna to maximize.
    """
    # Define hyperparameter search space
    hp = {
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'proj_hidden_dim': trial.suggest_categorical('proj_hidden_dim', [128, 256, 512, 768]),
        'proj_num_layers': trial.suggest_categorical('proj_num_layers', [1, 2, 3]),
        'normalize_cd': trial.suggest_categorical('normalize_cd', [True, False]),
        'learning_rate_src': trial.suggest_loguniform('learning_rate_src', 1e-7, 1e-3),
        'learning_rate_tgt': trial.suggest_loguniform('learning_rate_tgt', 1e-7, 1e-3),
        'learning_rate_fuse': trial.suggest_loguniform('learning_rate_fuse', 1e-6, 1e-2),
    }

    # Load base transfer config
    base_config_path = Path(f'config/transfer/{transfer_name}.yaml')
    if not base_config_path.exists():
        logging.error(f"Base transfer config not found at {base_config_path}")
        # Create a default one if it does not exist
        base_config = {'name': transfer_name, 'hp': {}}
    else:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

    # Create new transfer config for the trial
    trial_transfer_name = f"{transfer_name}_{trial.number}"
    trial_config = {
        'name': trial_transfer_name,
        'hp': hp
    }
    
    trial_config_path = Path(f'config/transfer/{trial_transfer_name}.yaml')
    with open(trial_config_path, 'w') as f:
        yaml.dump(trial_config, f, sort_keys=False)

    logging.info(f"Saved trial {trial.number} config to {trial_config_path}")

    # Construct command
    command = [
        sys.executable,
        'scripts/adapt.py',
        f'transfer={trial_transfer_name}',
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
            logging.error(f"Trial {trial.number}: No output from adapt.py.")
            return -1

        json_output = ''
        # Find the last line that is a valid json
        for line in reversed(output_lines):
            if line.strip().startswith('{') and line.strip().endswith('}'):
                json_output = line
                break
        
        if not json_output:
            logging.error(f"Trial {trial.number}: No JSON output from adapt.py. Full output: \n {process.stdout}")
            return -1

        metrics = json.loads(json_output)
        
        ndcg_10 = metrics.get('ndcg', {}).get('10', -1)
        logging.info(f"Trial {trial.number}: ndcg@10 = {ndcg_10}")

        if ndcg_10 == -1:
            logging.warning(f"Trial {trial.number}: 'ndcg' at 10 not found in metrics: {metrics}")

        return ndcg_10

    except subprocess.CalledProcessError as e:
        logging.error(f"Trial {trial.number}: adapt.py failed with exit code {e.returncode}.")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return -1 # Return a poor value
    except json.JSONDecodeError as e:
        logging.error(f"Trial {trial.number}: Failed to decode JSON from adapt.py output.")
        logging.error(f"Received output: {json_output}")
        return -1 # Return a poor value
    except Exception as e:
        logging.error(f"An unexpected error occurred during trial {trial.number}: {e}")
        return -1
    finally:
        # Clean up the generated config file
        if trial_config_path.exists():
            os.remove(trial_config_path)
            logging.info(f"Cleaned up trial config: {trial_config_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tuning/tune_adapt.py <transfer_name> [n_trials]")
        sys.exit(1)

    transfer_name = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    study_name = f'{transfer_name}-adapt-tuning'
    
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

    logging.info(f"Starting Optuna study '{study_name}' for transfer '{transfer_name}' with {n_trials} trials.")
    study.optimize(lambda trial: objective(trial, transfer_name), n_trials=n_trials)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print("      {}: {}".format(key, value))

    # Save the best hyperparameters to a yaml file
    best_config_path = Path(f'config/transfer/{transfer_name}_best_params.yaml')
    
    # Recreate the best config
    best_hp = study.best_trial.params.copy()
    
    best_config_content = {
        'name': transfer_name,
        'hp': best_hp
    }

    with open(best_config_path, 'w') as f:
        yaml.dump(best_config_content, f, sort_keys=False)
    
    logging.info(f"Saved best hyperparameters to {best_config_path}")
