from __future__ import absolute_import
import os
import sys
#from .osutils import mkdir_if_missing





class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout #Just make sure everything print on the screen, no different use "sys.stdout" and "sys.stderr".
        self.file = None
        if fpath is not None:
         #   mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


if __name__ == '__main__':
    #sys.stdout = Logger(fpath='test_log.txt')
    sys.stderr = Logger(fpath='error.txt')
    sys.stdout = Logger(fpath='log.txt')
    print("HELLssOssffuuuufs")
    a =0
    5/a