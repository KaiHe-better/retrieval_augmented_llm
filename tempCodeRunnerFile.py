# 假设你现在想要清空日志文件
# # 首先，移除现有的FileHandler
# logger.removeHandler(file_handler)
# file_handler.close()

# # 然后，清空文件内容
# with open(log_file, 'w') as file:
#     pass  # 打开文件后立即关闭，内容被清空

# # 最后，重新创建FileHandler并绑定到logger
# file_handler = logging.FileHandler(log_file)
# logger.addHandler(file_handler)

# # 继续写入新的日志
# logger.info('This is an info message after clearing the log.')