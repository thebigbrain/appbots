import time


class TimeConsuming:
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self):
        end_time = time.time()
        elapsed = int((end_time - self.start_time) * 1000) / 1000
        return elapsed, f"耗时 {elapsed} 秒"
