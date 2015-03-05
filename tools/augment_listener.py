from subprocess import Popen, PIPE, STDOUT
import re
import my_io
import time
import sys


ITERS_PER_EPOCH = 75
SLEEP_INTERVAL = 30

with open('/tmp/my_caffe_log.txt', 'r') as f:
    while True: 
        line = f.readline()
        if not line:
            break
    while True:
        line = f.readline()
        if not line: 
            time.sleep(SLEEP_INTERVAL)
        ii_re = re.search(r'Iteration (\d+), loss', line)
        if ii_re and int(ii_re.groups()[0]) % ITERS_PER_EPOCH == 0:
            print 'Approx 1 epoch has passed - time to re-augment'
            print line
            sys.stdout.flush()
            #my_io.create_aug_db()
            my_io.create_aug_lvl()
