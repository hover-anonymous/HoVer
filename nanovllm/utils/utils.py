import gc
import logging
import sys

# A context manager to control the state of the garbage collector.
# When entering the context, it sets the GC state to the specified state (enabled or disabled).
# When exiting the context, it restores the GC state to its previous state.
# also can be use as a decorator to wrap functions to control GC state during their execution.
class gc_control:
    def __init__(self, enable: bool):
        self.enable = enable
        self.prev_state = None

    def __enter__(self):
        self.prev_state = gc.isenabled()
        if self.enable:
            gc.enable()
        else:
            gc.disable()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev_state is not None:
            if self.prev_state:
                gc.enable()
            else:
                gc.disable()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# disable_gc: A decorator or context manager to disable garbage collection during the execution of a function or a block of code.
disable_gc = lambda enable=True: gc_control(enable=not enable)

# enable_gc: A decorator or context manager to enable garbage collection during the execution of a function or a block of code.
enable_gc = lambda enable=True: gc_control(enable=enable)

def setup_file_logger(name: str, log_file: str | None = None, level=logging.INFO):
    """
    Setup a logger that writes to stdout (always) and optionally to a file.
    Unified logging configuration with PID.
    """
    import logging
    import sys
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if logger.handlers:
        return logger
        
    # Standard format with Process ID
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s')

    # File Handler (Optional, only if filename provided)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file handler for {name}: {e}")
    
    # Stream Handler (Stdout) - Always add this so shell redirection works
    try:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    except Exception:
        pass
        
    return logger

