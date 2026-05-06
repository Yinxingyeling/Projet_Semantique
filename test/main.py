import subprocess
import sys

PIPELINE = [
    "embedding.py",
    "scores.py",
    "visualisation.py",
    "gold_manuel.py",
    "correlation.py"
]

for script in PIPELINE:
    print(f"Running {script}...")
    file = "test/" + script
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"Error while running {script}")
        sys.exit(result.returncode)