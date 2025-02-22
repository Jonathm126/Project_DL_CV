import webbrowser
import subprocess
import matplotlib.pyplot as plt
import time

# paths
import os
import sys

# tb open
logname = 'Feb-21_22-21-24'
log_dir =os.path.join(os.getcwd(),'models', 'model_logs', 'p_02',logname)

# tb launch
port = 6008
tb = subprocess.Popen(["tensorboard", f"--logdir={log_dir}", f"--port={port}", "--host=localhost"], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
webbrowser.open(f"http://localhost:{port}")