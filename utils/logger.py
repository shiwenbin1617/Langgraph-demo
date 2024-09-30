# coding: utf-8
import logging
import os
from colorama import Fore, init
from config import LOGGDIR

if not os.path.exists('./logs'):
    os.makedirs('./logs')


class Logger:
    _instances = {}

    def __new__(cls, name):
        if name not in cls._instances:
            cls._instances[name] = super(Logger, cls).__new__(cls)
            cls._instances[name]._initialized = False
        return cls._instances[name]

    def __init__(self, name):
        if not self._initialized:
            if os.path.exists(LOGGDIR):
                log_file_name = f"{LOGGDIR}rag_documentAI.log"
            else:
                log_file_name = "./logs/api_log.log"

            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False  # 防止日志传播到父级logger

            # 检查是否已经添加了处理器
            if not self.logger.handlers:
                # 创建文件处理器
                file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(file_handler)

                # 创建流处理器
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(stream_handler)

            init(autoreset=True)  # 初始化 colorama
            self._initialized = True

    def _log_with_color(self, level, message, color, emoji, fallback):
        emoji_text = emoji if self._supports_emoji() else fallback
        console_formatter = logging.Formatter(
            f"{color}%(asctime)s - %(name)s - %(levelname)s - {emoji_text} %(message)s")
        file_formatter = logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - {emoji_text} %(message)s")

        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(console_formatter)
            elif isinstance(handler, logging.FileHandler):
                handler.setFormatter(file_formatter)

        getattr(self.logger, level)(message)

        # 日志记录后重置格式化器
        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    def _supports_emoji(self):
        # 目前简单地返回 True，假设所有环境都支持 emoji
        return True

    def debug(self, message):
        self._log_with_color('debug', message, Fore.CYAN, '🐛', '[DEBUG]')

    def info(self, message):
        self._log_with_color('info', message, Fore.GREEN, 'ℹ️', '[INFO]')

    def warning(self, message):
        self._log_with_color('warning', message, Fore.YELLOW, '⚠️', '[WARNING]')

    def error(self, message):
        self._log_with_color('error', message, Fore.RED, '❌', '[ERROR]')

    def critical(self, message):
        self._log_with_color('critical', message, Fore.MAGENTA, '🔥', '[CRITICAL]')


# 使用示例
if __name__ == "__main__":
    # 禁用 root logger
    logging.getLogger().handlers = []

    logger = Logger(__name__)
    logger.debug("这是一条调试消息")
    logger.info("这是一条信息消息")
    logger.warning("这是一条警告消息")
    logger.error("这是一条错误消息")
    logger.critical("这是一条严重错误消息")


    # 测试在不同模块中使用
    def test_other_module():
        other_logger = Logger("other_module")
        other_logger.info("这是来自其他模块的消息")
    test_other_module()
