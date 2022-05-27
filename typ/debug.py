import time

DEBUG = False
t_locker = False
t = None
def debug_time_start():
    global DEBUG,t,t_locker
    if DEBUG:
        t = time.perf_counter()
        t_locker = True

def debug_print_time(str):
    global DEBUG,t,t_locker
    if DEBUG:
        if t_locker:
            print(str, f':{time.perf_counter() - t:.8f}s')
            t_locker = False
        else:
            print("没有记录开始时间")

def debug_print(*strs):
    global DEBUG
    if DEBUG:
        for str in strs:
            print(str, end="")
        print("")
