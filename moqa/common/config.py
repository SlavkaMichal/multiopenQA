import logging
from datetime import datetime

log_level = logging.DEBUG
timestamp = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
log_file = f"data/log/log_{timestamp}.txt" # None  # set to None for stdout
