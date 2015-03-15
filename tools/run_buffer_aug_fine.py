import os       
import glob                                                                                        
import subprocess
import my_io
import threading
import time
import make_fine_config as mfc

NDSB_DIR = '/media/raid_arr/data/ndsb/config/hier/'
# TRAIN_SCRIPT = os.path.join(NDSB_DIR, 'train_pl.sh')
# RESUME_SCRIPT = os.path.join(NDSB_DIR, 'resume_training_pl.sh')
SOLVER = os.path.join(NDSB_DIR, 'solver_coarse.prototxt')
NET = os.path.join(NDSB_DIR, 'cnn_v4_coarse.prototxt')
CAFFE = '/afs/ee.cooper.edu/user/t/a/tam8/documents/caffe/build/tools/caffe'
MODELS_DIR = '/media/raid_arr/data/ndsb/models'
BUFFER_PATH = '/media/raid_arr/tmp/aug_buffer/'
#~ snapshot_prefix = 'alex11_oriennormaugfeats_fold0_iter_'
snapshot_prefix = 'coarse_fold0_iter_'
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
    #if max_iter <= 2000:
    #    line_n = 10    # The line number we're going to replace
    #    new_line = 'momentum: ' + str(0.5 + 0.35*max_iter/2000) + '\n'
    #    f_data[line_n] = new_line

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

    ##########[SHELL COPY REMOVE =( ]##########
    dst_dir = '/dev/shm/'
    prefix = 'train0_aug_lvl'
    
    # Remove the last finished job from shm
    subprocess.call('rm -rf ' + dst_dir + prefix + '*', shell=True)
    
    # Copy new job to worksite
    for subpath in glob.glob(next_job_path + '*'):
        dst_name = os.path.basename(subpath).replace(next_job, prefix)
        dst = os.path.join(dst_dir, dst_name)
        subprocess.call(['cp', '-rf', subpath, dst])
    
        # Remove job from buffer
        #TODO: UNCOMMENT BELOW
        subprocess.call(['rm', '-rf', subpath])
    

    ## Don't need this if we have some other queue filling process in bckgnd
    # Start augmenting in another thread while caffe runs
    #aug_thread = threading.Thread(target=my_io.create_aug_lvl)
    #aug_thread.start()
    
    ####### COPYING COARSE ENTS TO FINE #####
    fine_solver_paths = mfc.copy_coarse_to_fine(NET, SOLVER)
   
    ################ CALLS TO CAFFE ###############
    if last_saved_iter == 0:
        print 'Starting new train'
        call_start()
        subprocess.call(['cp', '/tmp/caffe.INFO', 
                         '/tmp/my_caffe_log.txt'])
        
        #for ii, fine_solver_path in enumerate(fine_solver_paths):
        #    call_start(fine_solver_path)
        #    subprocess.call(['cp', '/tmp/caffe.INFO', 
        #                     '/tmp/my_caffe_log' + str(ii) + '.txt'])

    else:
        coarse_snap = snap_name(last_saved_iter)
        call_resume(coarse_snap)
        #####~ call_tune(weight_name(last_saved_iter))
        subprocess.call('tail -25 /tmp/caffe.INFO >> /tmp/my_caffe_log.txt', shell=True)
        
        #for ii, fine_solver_path in enumerate(fine_solver_paths):
        #    prefix = fine_solver_path.split('_')[-1].split('.prototxt')[0]
        #    snap_path = coarse_snap.replace('coarse', prefix)
        #    call_resume(snap_path , fine_solver_path)
        #    subprocess.call('tail -25 /tmp/caffe.INFO >> /tmp/my_caffe_log' + str(ii) + '.txt', shell=True)


print 'DONE'                        



