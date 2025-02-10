import logging


class RequestFormatter(logging.Formatter):
    """Custom formatter that includes a separator after each log message."""
    
    def format(self, record):
        formatted = super().format(record)
        return f"{formatted}\n{'-' * 80}"
