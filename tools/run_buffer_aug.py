import os                                                                                               
import subprocess
import my_io
import threading
import time
NDSB_DIR = '/media/raid_arr/data/ndsb/config'
# TRAIN_SCRIPT = os.path.join(NDSB_DIR, 'train_pl.sh')
# RESUME_SCRIPT = os.path.join(NDSB_DIR, 'resume_training_pl.sh')
#~ SOLVER = os.path.join(NDSB_DIR, 'solver.prototxt')
SOLVER = os.path.join(NDSB_DIR, 'solver_simple.prototxt')
#~ NET = os.path.join(NDSB_DIR, 'train_val.prototxt')
CAFFE = '/afs/ee.cooper.edu/user/t/a/tam8/documents/caffe/build/tools/caffe'
MODELS_DIR = '/media/raid_arr/data/ndsb/models'
BUFFER_PATH = '/media/raid_arr/tmp/aug_buffer/'
#~ snapshot_prefix = 'alex11_oriennormaugfeats_fold0_iter_'
snapshot_prefix = 'simple_fold0_iter_'
MAX_ITER = 50000    # global max (not per step)
STEP = 100      # MAKE SURE THE SNAPSHOT PARAM IN SOLVER MATCHES THIS



def write_max_iter_to_solver(max_iter, f_path=SOLVER):
    with open(f_path, 'r') as f:
    # read a list of lines into data
        f_data = f.readlines()

    # Change the line for maxiter (our stepsize)
    line_n = 8    # The line number we're going to replace
    new_line = 'max_iter: ' + str(max_iter) + '\n'
    f_data[line_n] = new_line
    
    # linearly scale up momentum
    if max_iter <= 1000:
        line_n = 10    # The line number we're going to replace
        new_line = 'momentum: ' + str(0.5 + 0.4*max_iter/1000) + '\n'
        f_data[line_n] = new_line

    # and write everything back
    with open(f_path, 'w') as f:
        f.writelines(f_data)
    return f_path



snap_name = lambda n_iter: os.path.join(MODELS_DIR,
            snapshot_prefix + str(n_iter) + '.solverstate')
            
weight_name = lambda n_iter: os.path.join(MODELS_DIR,
            snapshot_prefix + str(n_iter) + '.caffemodel')

call_start = lambda sol=SOLVER: subprocess.call(
            [CAFFE, 'train', '--solver=' + sol])
call_resume = lambda snap, sol=SOLVER: subprocess.call(
            [CAFFE, 'train', '--solver=' + sol, '--snapshot=' + snap])
call_tune = lambda weights, sol=SOLVER: subprocess.call(
            [CAFFE, 'train', '--solver=' + sol, '--weights=' + weights])



get_last_iter = lambda:  max({int(os.path.splitext(f)[0].rsplit('_', 1)[1])
                              for f in next(os.walk(MODELS_DIR))[2]}.union({0})) 

last_saved_iter = get_last_iter()  

#for ii in range(last_saved_iter, MAX_ITER+1, STEP):
last_saved_iter = get_last_iter()
while last_saved_iter < MAX_ITER:
    last_saved_iter = get_last_iter()  # lol
    write_max_iter_to_solver(last_saved_iter + STEP) # make caffe stop at the next step
    print 'ITER:\t', last_saved_iter
    
    # Get next job from ghetto queue
    jobs_int = None
    while not jobs_int:
        jobs_int = [int(p.split('_')[0]) for p in os.walk(BUFFER_PATH).next()[1]]
        if not jobs_int:
            print 'No jobs in queue! - waiting 10 sec'
            time.sleep(10)
    next_job = str(min(jobs_int))
    next_job_path = os.path.join(BUFFER_PATH, next_job)

    # Remove the last finished job
    subprocess.call(['rm', '-rf', '/dev/shm/train0_aug_lvl'])
    subprocess.call(['rm', '-rf', '/dev/shm/train0_aug_feats_lvl'])
    subprocess.call(['rm', '-rf', '/dev/shm/train0_aug_lbls_lvl'])
    
    # Copy new job to worksite
    subprocess.call(['cp', '-rf', next_job_path, '/dev/shm/train0_aug_lvl'])
    subprocess.call(['cp', '-rf', next_job_path + '_feats', '/dev/shm/train0_aug_feats_lvl'])
    subprocess.call(['cp', '-rf', next_job_path + '_lbls', '/dev/shm/train0_aug_lbls_lvl'])
    
    subprocess.call(['rm', '-rf', next_job_path])
    subprocess.call(['rm', '-rf', next_job_path + '_feats'])
    subprocess.call(['rm', '-rf', next_job_path + '_lbls'])

    ## Don't need this if we have some other queue filling process in bckgnd
    # Start augmenting in another thread while caffe runs
    #aug_thread = threading.Thread(target=my_io.create_aug_lvl)
    #aug_thread.start()
    
    if last_saved_iter == 0:
        print 'Starting new train'
        call_start()
        subprocess.call(['cp', '/tmp/caffe.INFO', 
                         '/tmp/my_caffe_log.txt'])
    else:
        call_resume(snap_name(last_saved_iter))
        #~ call_tune(weight_name(last_saved_iter))
        subprocess.call(['tail -25 /tmp/caffe.INFO >> /tmp/my_caffe_log.txt'], shell=True)
print 'DONE'                        



