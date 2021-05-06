import os
import socket
import logging
from datetime import datetime

log_level = logging.DEBUG
timestamp = datetime.now().strftime("M%mD%d_%H_%M_%S")
log_dir = os.environ.get('log_dir', 'data/logs')
log_suffix = os.environ.get('log_file_suffix', '')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
log_file = f"{log_dir}/log_{timestamp}_{socket.gethostname()}_{log_suffix}.txt"  # None  # set to None for stdout
print(f'logging to file: {os.path.abspath(log_file)}')
