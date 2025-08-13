import subprocess
import time

def get_gpu_stats():
    try:
        # Query nvidia-smi for GPU utilization and memory info
        result = subprocess.run(
            ["nvidia-smi", 
             "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu", 
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        stats = []
        for line in lines:
            idx, name, mem_total, mem_used, util, temp = line.split(', ')
            stats.append({
                "index": idx,
                "name": name,
                "mem_total": int(mem_total),
                "mem_used": int(mem_used),
                "utilization": int(util),
                "temperature": int(temp),
            })
        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def print_stats(stats):
    print(f"{'GPU':>3} | {'Name':<25} | {'Mem Used / Total (MB)':>20} | {'Util %':>6} | {'Temp (°C)':>8}")
    print("-" * 75)
    for s in stats:
        print(f"{s['index']:>3} | {s['name']:<25} | {s['mem_used']:6} / {s['mem_total']:6}       | {s['utilization']:6}% | {s['temperature']:8}°C")
    print()

if __name__ == "__main__":
    try:
        while True:
            gpu_stats = get_gpu_stats()
            print_stats(gpu_stats)
            time.sleep(5)  # update every 5 seconds
    except KeyboardInterrupt:
        print("Monitoring stopped.")
