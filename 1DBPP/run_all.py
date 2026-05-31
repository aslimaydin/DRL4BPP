"""
run_all.py
==========
3 GNN × 5 Algoritma = 15 deneyi sırayla eğitir.

Özellikler:
- Zaten tamamlanan deneyleri atlar (checkpoint'e bakarak)
- Her deneyin sonucunu summary_results.json'a yazar
- İstenirse belirli GNN veya algoritma filtreleri uygulanabilir

Kullanım:
    python run_all.py                        # Tüm 15 deney
    python run_all.py --gnn gat              # Sadece GAT deneyleri
    python run_all.py --alg ppo reinforce    # Sadece PPO ve REINFORCE
    python run_all.py --epochs 1000          # Epoch sayısını değiştir
    python run_all.py --skip_done            # Tamamlananları atla (varsayılan)
    python run_all.py --force                # Hepsini sıfırdan eğit
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta

# Ensure this process uses UTF-8 output
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# KONFİGÜRASYON
# ─────────────────────────────────────────────────────────────────────────────

GNN_TYPES  = ['gcn', 'gat', 'gin']
ALGORITHMS = ['reinforce', 'a2c', 'ppo', 'dqn', 'sac']

# Varsayılan eğitim parametreleri
DEFAULT_EPOCHS     = 2000
DEFAULT_BATCH_SIZE = 16
DEFAULT_N_ITEMS    = 50
DEFAULT_CAPACITY   = 100
DEFAULT_REWARD     = 'step'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistematik GNN × Algoritma Toplu Eğitim"
    )
    parser.add_argument('--gnn', nargs='+', choices=GNN_TYPES,
                        default=GNN_TYPES,
                        help='Eğitilecek GNN tipleri (varsayılan: hepsi)')
    parser.add_argument('--alg', nargs='+', choices=ALGORITHMS,
                        default=ALGORITHMS,
                        help='Eğitilecek algoritmalar (varsayılan: hepsi)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Epoch sayısı (varsayılan: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch boyutu (varsayılan: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--n_items', type=int, default=DEFAULT_N_ITEMS,
                        help=f'Item sayısı (varsayılan: {DEFAULT_N_ITEMS})')
    parser.add_argument('--capacity', type=int, default=DEFAULT_CAPACITY,
                        help=f'Bin kapasitesi (varsayılan: {DEFAULT_CAPACITY})')
    parser.add_argument('--reward', type=str, default=DEFAULT_REWARD,
                        choices=['step', 'terminal'],
                        help=f'Reward tipi (varsayılan: {DEFAULT_REWARD})')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Tamamlanmış deneyleri de yeniden eğit')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='GPU kullan')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Sadece planı göster, eğitim yapma')
    parser.add_argument('--summary_file', type=str,
                        default='summary_results.json',
                        help='Summary output file')
    parser.add_argument('-y', '--yes', action='store_true', default=False,
                        help='Auto-confirm without ENTER prompt')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

def is_done(exp_name: str) -> bool:
    """Deneyin tamamlanıp tamamlanmadığını kontrol et."""
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    final_model    = os.path.join(checkpoint_dir, 'final_model.pth')
    training_log   = os.path.join(checkpoint_dir, 'training_log.json')
    return os.path.exists(final_model) and os.path.exists(training_log)


def get_best_val_bins(exp_name: str) -> float:
    """Training log'dan en iyi validation bins değerini oku."""
    log_path = os.path.join('checkpoints', exp_name, 'training_log.json')
    if not os.path.exists(log_path):
        return float('inf')
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
        if log:
            return min(entry.get('avg_bins', float('inf')) for entry in log[-100:])
    except Exception:
        pass
    return float('inf')


def format_time(seconds: float) -> str:
    """Saniyeyi okunabilir formata çevir."""
    return str(timedelta(seconds=int(seconds)))


def build_command(gnn: str, alg: str, args) -> list:
    """rl_train.py için komut satırı argümanlarını oluştur."""
    exp_name = f"{gnn}_{alg}_{args.reward}"
    cmd = [
        sys.executable, 'rl_train.py',
        '--gnn_type',   gnn,
        '--algorithm',  alg,
        '--reward_type', args.reward,
        '--epochs',     str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--n_items',    str(args.n_items),
        '--capacity',   str(args.capacity),
        '--experiment_name', exp_name,
    ]
    if args.gpu:
        cmd.append('--gpu')
    return cmd, exp_name


def print_header():
    print("\n" + "=" * 70)
    print("  SYSTEMATIC GNN x ALGORITHM TRAINING PLAN")
    print("=" * 70)


def print_plan(experiments, args):
    """Show which experiments will run."""
    print(f"\n{'Experiment':<30} {'Status':<15} {'GNN':<6} {'Algorithm':<12}")
    print("-" * 65)
    for exp_name, gnn, alg, skip in experiments:
        status = "[DONE]" if skip else "[RUN]"
        print(f"{exp_name:<30} {status:<15} {gnn.upper():<6} {alg.upper():<12}")
    
    n_todo = sum(1 for _, _, _, skip in experiments if not skip)
    n_done = sum(1 for _, _, _, skip in experiments if skip)
    print(f"\nTotal: {len(experiments)} experiments | To run: {n_todo} | Skip: {n_done}")
    print(f"Est. time: ~{n_todo * args.epochs * 2 / 60:.0f} min (~2s per epoch)")


# ─────────────────────────────────────────────────────────────────────────────
# ANA EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    print_header()
    print(f"\nParameters:")
    print(f"  GNN types    : {args.gnn}")
    print(f"  Algorithms   : {args.alg}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  N items      : {args.n_items}")
    print(f"  Capacity     : {args.capacity}")
    print(f"  Reward       : {args.reward}")
    print(f"  GPU          : {args.gpu}")
    print(f"  Force retrain: {args.force}")
    
    # Deney listesini hazırla
    experiments = []
    for gnn in args.gnn:
        for alg in args.alg:
            cmd, exp_name = build_command(gnn, alg, args)
            already_done = is_done(exp_name) and not args.force
            experiments.append((exp_name, gnn, alg, already_done, cmd))
    
    print_plan([(e, g, a, s) for e, g, a, s, _ in experiments], args)
    
    if args.dry_run:
        print("\n[DRY RUN] No training started.")
        return
    
    # Kullanıcı onayı
    n_todo = sum(1 for _, _, _, skip, _ in experiments if not skip)
    if n_todo == 0:
        print("\nAll experiments already completed!")
        return
    
    print(f"\n{n_todo} experiments will be trained. Press ENTER to start (Ctrl+C to cancel)...")
    if not args.yes:
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return
    
    # Özet sonuçları yükle
    summary_path = args.summary_file
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # ─── EĞİTİM DÖNGÜSÜ ───
    total_start = time.time()
    completed   = 0
    failed      = []
    
    for i, (exp_name, gnn, alg, skip, cmd) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiments)}] {exp_name.upper()}")
        print(f"{'='*70}")
        
        if skip:
            best_bins = get_best_val_bins(exp_name)
            print(f"  [SKIP] Already done | Best bins: {best_bins:.1f}")
            continue
        
        print(f"  GNN: {gnn.upper()} | Algorithm: {alg.upper()}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        exp_start = time.time()
        
        try:
            # Pass UTF-8 encoding to child processes
            child_env = os.environ.copy()
            child_env['PYTHONUTF8'] = '1'
            child_env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=False,
                env=child_env,
                timeout=args.epochs * 30,
            )
            
            exp_time = time.time() - exp_start
            
            if result.returncode == 0:
                best_bins = get_best_val_bins(exp_name)
                print(f"\n  [OK] {exp_name} done | "
                      f"Time: {format_time(exp_time)} | "
                      f"Best bins: {best_bins:.1f}")
                
                summary[exp_name] = {
                    'gnn':      gnn,
                    'alg':      alg,
                    'status':   'completed',
                    'time_s':   round(exp_time, 1),
                    'best_bins': best_bins,
                    'timestamp': datetime.now().isoformat(),
                }
                completed += 1
            else:
                print(f"\n  [FAIL] {exp_name} ERROR (returncode={result.returncode})")
                summary[exp_name] = {
                    'gnn':    gnn,
                    'alg':    alg,
                    'status': 'failed',
                    'returncode': result.returncode,
                    'timestamp': datetime.now().isoformat(),
                }
                failed.append(exp_name)
        
        except subprocess.TimeoutExpired:
            exp_time = time.time() - exp_start
            print(f"\n  [TIMEOUT] {exp_name} ({format_time(exp_time)})")
            summary[exp_name] = {
                'gnn': gnn, 'alg': alg, 'status': 'timeout',
                'timestamp': datetime.now().isoformat(),
            }
            failed.append(exp_name)
        
        except KeyboardInterrupt:
            print(f"\n\n[STOPPED] User interrupted ({exp_name} was running).")
            break
        
        except Exception as e:
            print(f"\n  [ERROR] {exp_name}: {e}")
            summary[exp_name] = {
                'gnn': gnn, 'alg': alg, 'status': 'error', 'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
            failed.append(exp_name)
        
        finally:
            # Her deneyin ardından özeti güncelle
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # ─── ÖZET RAPOR ───
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time   : {format_time(total_time)}")
    print(f"  Completed    : {completed}")
    print(f"  Failed       : {len(failed)}")
    
    if summary:
        print(f"\n{'Experiment':<30} {'GNN':<6} {'Alg':<12} {'Status':<12} {'Best Bins':<12}")
        print("-" * 75)
        for exp_name, info in sorted(summary.items()):
            bins_str = f"{info['best_bins']:.1f}" if 'best_bins' in info else "-"
            print(f"{exp_name:<30} {info['gnn'].upper():<6} "
                  f"{info['alg'].upper():<12} {info['status']:<12} {bins_str:<12}")
    
    if failed:
        print(f"\n  Failed experiments: {failed}")
    
    print(f"\n  Summary saved: {summary_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
