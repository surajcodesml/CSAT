"""
JSON logging utility for experiment tracking.
Logs hyperparameters and per-epoch metrics to JSON files.
"""
import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Logs experiment configuration and training metrics to JSON."""
    
    def __init__(self, experiment_name, config, log_dir='logs'):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{experiment_name}_{timestamp}.json'
        
        # Initialize log structure
        self.log_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': config,
            'epochs': []
        }
        
        # Save initial config
        self._save()
    
    def log_epoch(self, epoch, metrics):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics (loss, lr, val_metrics, etc.)
        """
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.log_data['epochs'].append(epoch_data)
        self._save()
    
    def log_final_results(self, results):
        """
        Log final experiment results.
        
        Args:
            results: Dictionary of final metrics
        """
        self.log_data['final_results'] = results
        self._save()
    
    def _save(self):
        """Save log data to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def get_log_path(self):
        """Return path to the log file."""
        return str(self.log_file)


def load_experiment_log(log_file):
    """
    Load experiment log from JSON file.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Dictionary with experiment data
    """
    with open(log_file, 'r') as f:
        return json.load(f)


def compare_experiments(log_files):
    """
    Compare multiple experiment logs.
    
    Args:
        log_files: List of log file paths
        
    Returns:
        Comparison dictionary
    """
    experiments = []
    for log_file in log_files:
        data = load_experiment_log(log_file)
        
        # Extract key metrics
        final_epoch = data['epochs'][-1] if data['epochs'] else {}
        summary = {
            'name': data['experiment_name'],
            'final_epoch': final_epoch.get('epoch', 0),
            'final_train_loss': final_epoch.get('train_loss', None),
            'final_val_loss': final_epoch.get('val_loss', None),
            'best_fitness': final_epoch.get('fitness', None),
            'config': data['config']
        }
        
        if 'final_results' in data:
            summary['final_results'] = data['final_results']
        
        experiments.append(summary)
    
    return experiments
