#!/usr/bin/env python3
"""
Real-time training monitor for the Master Thesis model.
Tracks progress, ETA, and metrics.
"""
import os
import time
import re
from datetime import datetime, timedelta

LOG_FILE = r"C:\Users\Bruger\PycharmProjects\MasterThesis\training.log"

def parse_log_line(line):
    """Extract epoch, rmse, and other metrics from log line."""
    match = re.search(r'epoch=\s*(\d+).*train_rmse=([\d.e+-]+).*avg_rmse_bps=([\d.]+).*lr=([\d.e+-]+).*time_total=([\d.]+)min', line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'rmse': float(match.group(2)),
            'rmse_bps': float(match.group(3)),
            'lr': float(match.group(4)),
            'time_min': float(match.group(5)),
        }
    return None

def monitor_training(check_interval=10, total_epochs=500):
    """Monitor training with periodic updates."""
    last_epoch = -1
    start_time = datetime.now()
    
    print(f"🚀 Starting training monitor at {start_time.strftime('%H:%M:%S')}")
    print(f"📊 Target: {total_epochs} epochs")
    print(f"🔄 Checking every {check_interval} seconds\n")
    print("-" * 100)
    
    while True:
        try:
            if not os.path.exists(LOG_FILE):
                print(f"⏳ Waiting for log file: {LOG_FILE}")
                time.sleep(check_interval)
                continue
            
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
            
            # Find last epoch line
            last_entry = None
            for line in reversed(lines):
                parsed = parse_log_line(line)
                if parsed:
                    last_entry = parsed
                    break
            
            if last_entry and last_entry['epoch'] != last_epoch:
                epoch = last_entry['epoch']
                progress = (epoch / total_epochs) * 100
                elapsed = timedelta(minutes=last_entry['time_min'])
                
                # Calculate ETA
                if epoch > 0:
                    time_per_epoch = last_entry['time_min'] / epoch
                    remaining_epochs = total_epochs - epoch
                    eta_min = time_per_epoch * remaining_epochs
                    eta_time = datetime.now() + timedelta(minutes=eta_min)
                else:
                    eta_min = 0
                    eta_time = None
                
                last_epoch = epoch
                
                # Print update
                print(f"⏱️  {datetime.now().strftime('%H:%M:%S')} | "
                      f"Epoch {epoch:3d}/{total_epochs} ({progress:5.1f}%) | "
                      f"RMSE: {last_entry['rmse']:.4e} ({last_entry['rmse_bps']:6.2f} bps) | "
                      f"LR: {last_entry['lr']:.2e} | "
                      f"⏳ {elapsed} | "
                      f"ETA: {eta_time.strftime('%H:%M:%S') if eta_time else 'calculating...'}")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\n⛔ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    monitor_training(check_interval=10, total_epochs=500)

