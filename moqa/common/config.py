import logging
from datetime import datetime

log_level = logging.DEBUG
timestamp = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
log_file = None  # f"../../data/log/{timestamp}" # None  # set to None for stdout
