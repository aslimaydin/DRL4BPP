"""
run_all_2d.py
=============

3 GNN x 5 Algorithm = 15 experiments for 2D BPP.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

GNN_TYPES  = ['gcn', 'gat', 'gin']
ALGORITHMS = ['reinforce', 'a2c', 'ppo', 'dqn', 'sac']

DEFAULT_EPOCHS     = 2000
DEFAULT_BATCH_SIZE = 16
DEFAULT_N_ITEMS    = 20
DEFAULT_BIN_W      = 100
DEFAULT_BIN_H      = 100
DEFAULT_REWARD     = 'step'


def parse_args():
    parser = argparse.ArgumentParser(description="2D BPP Batch Training")
    parser.add_argument('--gnn', nargs='+', choices=GNN_TYPES, default=GNN_TYPES)
    parser.add_argument('--alg', nargs='+', choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--n_items', type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument('--bin_width', type=int, default=DEFAULT_BIN_W)
    parser.add_argument('--bin_height', type=int, default=DEFAULT_BIN_H)
    parser.add_argument('--reward', type=str, default=DEFAULT_REWARD)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('-y', '--yes', action='store_true', default=False)
    parser.add_argument('--summary_file', type=str, default='summary_results_2d.json')
    return parser.parse_args()


def is_done(exp_name):
    d = os.path.join('checkpoints_2d', exp_name)
    return (os.path.exists(os.path.join(d, 'final_model.pth')) and
            os.path.exists(os.path.join(d, 'training_log.json')))


def get_best_val(exp_name):
    log_path = os.path.join('checkpoints_2d', exp_name, 'training_log.json')
    if not os.path.exists(log_path):
        return float('inf')
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
        if log:
            vals = [e.get('model_avg_groups', float('inf')) for e in log[-100:]]
            return min(vals)
    except Exception:
        pass
    return float('inf')


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 70)
    print("  2D BPP: SYSTEMATIC GNN x ALGORITHM TRAINING")
    print("=" * 70)
    print(f"  GNN types : {args.gnn}")
    print(f"  Algorithms: {args.alg}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Bin       : {args.bin_width}x{args.bin_height}")
    print(f"  N items   : {args.n_items}")
    print(f"  GPU       : {args.gpu}")

    experiments = []
    for gnn in args.gnn:
        for alg in args.alg:
            exp_name = f"{gnn}_{alg}_{args.reward}"
            done = is_done(exp_name) and not args.force
            cmd = [
                sys.executable, os.path.join(script_dir, 'rl_train_2d.py'),
                '--gnn_type', gnn, '--algorithm', alg,
                '--reward_type', args.reward,
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--n_items', str(args.n_items),
                '--bin_width', str(args.bin_width),
                '--bin_height', str(args.bin_height),
                '--experiment_name', exp_name,
            ]
            if args.gpu:
                cmd.append('--gpu')
            experiments.append((exp_name, gnn, alg, done, cmd))

    # Print plan
    print(f"\n{'Experiment':<30} {'Status':<12}")
    print("-" * 45)
    n_todo = 0
    for exp_name, gnn, alg, skip, _ in experiments:
        status = "[DONE]" if skip else "[RUN]"
        print(f"{exp_name:<30} {status:<12}")
        if not skip:
            n_todo += 1

    print(f"\nTo run: {n_todo} / {len(experiments)}")

    if args.dry_run or n_todo == 0:
        return

    if not args.yes:
        input(f"\nPress ENTER to start (Ctrl+C to cancel)...")

    # Load summary
    if os.path.exists(args.summary_file):
        with open(args.summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}

    total_start = time.time()
    completed = 0
    failed = []

    child_env = os.environ.copy()
    child_env['PYTHONUTF8'] = '1'
    child_env['PYTHONIOENCODING'] = 'utf-8'

    for i, (exp_name, gnn, alg, skip, cmd) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiments)}] {exp_name.upper()}")
        print(f"{'='*70}")

        if skip:
            print(f"  [SKIP] Already done | Best: {get_best_val(exp_name):.1f}")
            continue

        exp_start = time.time()
        try:
            result = subprocess.run(
                cmd, cwd=script_dir, capture_output=False,
                env=child_env, timeout=args.epochs * 30
            )
            exp_time = time.time() - exp_start

            if result.returncode == 0:
                best = get_best_val(exp_name)
                print(f"\n  [OK] {exp_name} | Time: {format_time(exp_time)} | "
                      f"Best: {best:.1f}")
                summary[exp_name] = {
                    'gnn': gnn, 'alg': alg, 'status': 'completed',
                    'time_s': round(exp_time, 1), 'best_groups': best,
                    'timestamp': datetime.now().isoformat(),
                }
                completed += 1
            else:
                print(f"\n  [FAIL] {exp_name} (returncode={result.returncode})")
                summary[exp_name] = {
                    'gnn': gnn, 'alg': alg, 'status': 'failed',
                    'returncode': result.returncode,
                    'timestamp': datetime.now().isoformat(),
                }
                failed.append(exp_name)

        except subprocess.TimeoutExpired:
            print(f"\n  [TIMEOUT] {exp_name}")
            failed.append(exp_name)
        except KeyboardInterrupt:
            print(f"\n\n[STOPPED] User interrupted.")
            break
        except Exception as e:
            print(f"\n  [ERROR] {exp_name}: {e}")
            failed.append(exp_name)
        finally:
            with open(args.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Completed : {completed}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print(f"  Failed list: {failed}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
