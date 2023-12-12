import logging

# 设定日志文件
log_file = 'example.log'

# 配置logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

# 写入一些日志
logger.debug('This is a debug message.')
logger.info('This is an info message before clearing the log.')

# 假设你现在想要清空日志文件
# 首先，移除现有的FileHandler
file_handler = logger.handlers[0]
log_file = logger.handlers[0].baseFilename
logger.removeHandler(file_handler)
file_handler.close()

# 然后，清空文件内容
with open(log_file, 'w') as file:
    pass  # 打开文件后立即关闭，内容被清空

# 最后，重新创建FileHandler并绑定到logger
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

# 继续写入新的日志
logger.info('This is an info message after clearing the log.')