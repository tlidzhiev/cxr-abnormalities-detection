import logging
import logging.config
from pathlib import Path


def setup_logging(
    log_dir: Path,
    level: int = logging.INFO,
    append: bool = False,
) -> None:
    """
    Configure the root logger to print to the console and to 'log_dir/info.log'.

    Parameters
    ----------
    log_dir : Path
        Directory of the run.
    level : int, optional
        Logging level.
    append : bool, optional
        Append to an existing log file (used on resume).
    """
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {'format': '%(message)s'},
                'datetime': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'},
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'datetime',
                    'stream': 'ext://sys.stdout',
                },
                'info_file': {
                    'class': 'logging.FileHandler',
                    'formatter': 'datetime',
                    'filename': str(log_dir / 'info.log'),
                    'mode': 'a' if append else 'w',
                    'encoding': 'utf8',
                },
            },
            'root': {'level': level, 'handlers': ['console', 'info_file']},
        }
    )
