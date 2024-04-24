import logging
import datetime as dt

def initialize_logger(config, file_name):
    # Clear handlers if they were created in other runs
    logging.getLogger().handlers = []

    # Set up the logger
    log_name = '../out_logs/coteaching/' + file_name + '.log'
    logging.basicConfig(filename=log_name, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    """
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    """

    logger = logging.getLogger()
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        logger.debug("{} -->   {}".format(keystr, val))
    
    return logger
