import logging

class Logger:
    def __init__(self, log_file_name):
        self.logger = logging.getLogger('pipeline_logger')
        
        # Set the log level
        self.logger.setLevel(logging.DEBUG)
        
        # Create a file handler to write to the logfile
        file_handler = logging.FileHandler(log_file_name)
    
        # Create a formatter to format the log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Add the formatter to the file handler
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
        
    def get_logger(self):
        return self.logger
        
    def log(self, message, level="INFO"):
        if level == "INFO":
            self.logger.info(message)
        elif level == "DEBUG":
            self.logger.debug(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
        else:
            self.logger.debug(message)