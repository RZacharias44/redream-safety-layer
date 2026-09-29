import logging
import sys

def setup_logging():
    # Base configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Set prompt logger level
    prompt_logger = logging.getLogger("prompts")
    prompt_logger.setLevel(logging.INFO)  # or logging.INFO, logging.WARNING, etc.
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Configure prompt logger first
    prompt_logger.addHandler(console_handler)
    prompt_logger.propagate = False  # Prevent propagation to root logger
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

