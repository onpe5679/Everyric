import logging
from typing import Optional
import sys

class ColoredFormatter(logging.Formatter):
    """로거에 색상 추가"""
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 중복 핸들러 방지
    if not logger.handlers:
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
    
    return logger

class log_step:
    """단계별 로깅을 위한 컨텍스트 매니저"""
    def __init__(self, step: str, logger: Optional[logging.Logger] = None):
        self.step = step
        self.logger = logger or logging.getLogger(__name__)
        
    def __enter__(self):
        self.logger.info(f"🚀 {self.step} 시작")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"✅ {self.step} 완료")
        else:
            self.logger.error(f"❌ {self.step} 실패: {str(exc_val)}", exc_info=True)
        return False  # 예외를 다시 발생시킴
