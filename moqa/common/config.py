import os.path

import logging
from datetime import datetime

log_level = logging.DEBUG
timestamp = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
log_dir = 'data/logs'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
log_file = f"{log_dir}/log_{timestamp}.txt"  # None  # set to None for stdout
print(f'logging to file: {os.path.abspath(log_file)}')
