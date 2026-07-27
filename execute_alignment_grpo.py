import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = "/home/lmassaron/code/sft-examples/alignment_grpo.ipynb"
output_log = "/home/lmassaron/code/sft-examples/alignment_grpo_execution.log"

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}\n"
    print(line, end="", flush=True)
    with open(output_log, "a") as f:
        f.write(line)

log(f"Starting detached execution of {notebook_path}...")

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=14400, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "/home/lmassaron/code/sft-examples"}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    log(f"SUCCESS: Notebook executed and updated at {notebook_path}")

except Exception as e:
    log(f"ERROR during notebook execution: {e}")
    sys.exit(1)
