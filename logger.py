from datetime import datetime


def log(string):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    log_string = "[" + current_time + "]" + str(string)
    print(log_string)
