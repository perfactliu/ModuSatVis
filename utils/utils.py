import sys, os


def resource_path(relative_path):
    """打包后/开发时都能读取资源文件"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
