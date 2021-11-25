def epoch_time(time, curr_epoch, total_epochs):
    minutes = int(time / 60)
    seconds = int(time % 60)

    epoch_left = total_epochs - curr_epoch
    time_left = epoch_left * time
    time_left_min = int(time_left / 60) - minutes
    time_left_sec = int(time_left % 60)

    return minutes, seconds, time_left_min, time_left_sec    