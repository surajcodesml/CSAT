"""
Compare results from multiple CSAT experiments.
Analyzes JSON log files and generates comparison report.
"""
import json
import sys
from pathlib import Path
from tabulate import tabulate


def load_experiment(log_file):
    """Load experiment data from JSON log file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def extract_key_metrics(exp_data):
    """Extract key metrics from experiment data."""
    exp_name = exp_data['experiment_name']
    
    # Get configuration
    config = exp_data['config']
    lr = config['training']['learning_rate']
    batch_size = config['training']['train_batch_size']
    num_queries = config['model']['num_queries']
    dropout = config['model']['dropout']
    
    # Analyze epochs
    if not exp_data['epochs']:
        return None
    
    epochs_data = exp_data['epochs']
    first_epoch = epochs_data[0]
    last_epoch = epochs_data[-1]
    
    # Find best fitness epoch
    best_epoch_idx = max(range(len(epochs_data)), key=lambda i: epochs_data[i].get('fitness', 0))
    best_epoch = epochs_data[best_epoch_idx]
    
    # Calculate loss improvement
    initial_train_loss = first_epoch.get('train_loss', 0)
    final_train_loss = last_epoch.get('train_loss', 0)
    loss_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss * 100) if initial_train_loss > 0 else 0
    
    # Check if loss is decreasing (compare first 3 and last 3 epochs)
    early_loss = sum(e.get('train_loss', 0) for e in epochs_data[:3]) / min(3, len(epochs_data))
    late_loss = sum(e.get('train_loss', 0) for e in epochs_data[-3:]) / min(3, len(epochs_data))
    is_learning = early_loss > late_loss
    
    return {
        'name': exp_name,
        'num_epochs': len(epochs_data),
        'lr': lr,
        'batch_size': batch_size,
        'num_queries': num_queries,
        'dropout': dropout,
        'initial_train_loss': initial_train_loss,
        'final_train_loss': final_train_loss,
        'loss_improvement_pct': loss_improvement,
        'best_fitness': best_epoch.get('fitness', 0),
        'best_fitness_epoch': best_epoch.get('epoch', 0),
        'final_val_loss': last_epoch.get('val_loss', 0),
        'is_learning': is_learning,
        'final_lr': last_epoch.get('learning_rate', lr)
    }


def compare_experiments(log_dir='logs'):
    """Compare all experiments in the log directory."""
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob('exp*.json'))
    
    if not json_files:
        print(f"No experiment JSON files found in {log_dir}/")
        return
    
    print(f"\nFound {len(json_files)} experiment log files:\n")
    
    experiments = []
    for log_file in json_files:
        try:
            exp_data = load_experiment(log_file)
            metrics = extract_key_metrics(exp_data)
            if metrics:
                experiments.append(metrics)
                print(f"  ✓ Loaded: {log_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {log_file.name}: {e}")
    
    if not experiments:
        print("\nNo valid experiment data found.")
        return
    
    # Sort by best fitness (descending)
    experiments_sorted = sorted(experiments, key=lambda x: x['best_fitness'], reverse=True)
    
    # Prepare comparison table
    print("\n" + "="*120)
    print("EXPERIMENT COMPARISON - Sorted by Best Fitness")
    print("="*120 + "\n")
    
    # Configuration table
    config_headers = ['Experiment', 'LR', 'Batch', 'Queries', 'Dropout', 'Epochs']
    config_rows = [[e['name'], f"{e['lr']:.6f}", e['batch_size'], e['num_queries'], 
                    e['dropout'], e['num_epochs']] for e in experiments_sorted]
    
    print("Configuration:")
    print(tabulate(config_rows, headers=config_headers, tablefmt='grid'))
    
    # Performance table
    print("\n\nPerformance Metrics:")
    perf_headers = ['Experiment', 'Best Fitness', 'Best @ Epoch', 'Initial Loss', 
                    'Final Loss', 'Improvement %', 'Learning?']
    perf_rows = []
    for e in experiments_sorted:
        learning_status = '✓ Yes' if e['is_learning'] else '✗ No'
        perf_rows.append([
            e['name'],
            f"{e['best_fitness']:.4f}",
            e['best_fitness_epoch'],
            f"{e['initial_train_loss']:.4f}",
            f"{e['final_train_loss']:.4f}",
            f"{e['loss_improvement_pct']:.1f}%",
            learning_status
        ])
    
    print(tabulate(perf_rows, headers=perf_headers, tablefmt='grid'))
    
    # Recommendations
    print("\n\n" + "="*120)
    print("RECOMMENDATIONS")
    print("="*120 + "\n")
    
    best_exp = experiments_sorted[0]
    learning_exps = [e for e in experiments_sorted if e['is_learning']]
    
    print(f"🏆 Best Overall Performance: {best_exp['name']}")
    print(f"   - Best fitness: {best_exp['best_fitness']:.4f}")
    print(f"   - Learning rate: {best_exp['lr']:.6f}")
    print(f"   - Batch size: {best_exp['batch_size']}")
    print(f"   - Num queries: {best_exp['num_queries']}")
    print(f"   - Dropout: {best_exp['dropout']}")
    
    if learning_exps:
        print(f"\n✓ Experiments showing learning ({len(learning_exps)}/{len(experiments_sorted)}):")
        for e in learning_exps[:3]:  # Show top 3
            print(f"   - {e['name']}: Loss improved by {e['loss_improvement_pct']:.1f}%")
    
    not_learning = [e for e in experiments_sorted if not e['is_learning']]
    if not_learning:
        print(f"\n✗ Experiments NOT learning ({len(not_learning)}/{len(experiments_sorted)}):")
        for e in not_learning[:3]:  # Show top 3
            print(f"   - {e['name']}: Loss changed by {e['loss_improvement_pct']:.1f}%")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    avg_fitness = sum(e['best_fitness'] for e in experiments_sorted) / len(experiments_sorted)
    avg_improvement = sum(e['loss_improvement_pct'] for e in experiments_sorted) / len(experiments_sorted)
    print(f"   - Average best fitness: {avg_fitness:.4f}")
    print(f"   - Average loss improvement: {avg_improvement:.1f}%")
    print(f"   - Learning rate range: {min(e['lr'] for e in experiments_sorted):.6f} - {max(e['lr'] for e in experiments_sorted):.6f}")
    print(f"   - Batch size range: {min(e['batch_size'] for e in experiments_sorted)} - {max(e['batch_size'] for e in experiments_sorted)}")
    
    print("\n" + "="*120 + "\n")
    
    # Save detailed comparison to file
    output_file = Path(log_dir) / 'comparison_report.txt'
    with open(output_file, 'w') as f:
        f.write("EXPERIMENT COMPARISON REPORT\n")
        f.write("="*120 + "\n\n")
        f.write(tabulate(config_rows, headers=config_headers, tablefmt='grid'))
        f.write("\n\n")
        f.write(tabulate(perf_rows, headers=perf_headers, tablefmt='grid'))
    
    print(f"Detailed report saved to: {output_file}\n")


if __name__ == '__main__':
    # Check if tabulate is available
    try:
        import tabulate
    except ImportError:
        print("ERROR: 'tabulate' package not found.")
        print("Install with: pip install tabulate")
        sys.exit(1)
    
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    compare_experiments(log_dir)
