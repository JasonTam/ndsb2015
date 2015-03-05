import os                                                                                               
import subprocess
import my_io
import threading
NDSB_DIR = '/media/raid_arr/data/ndsb/config'
# TRAIN_SCRIPT = os.path.join(NDSB_DIR, 'train_pl.sh')
# RESUME_SCRIPT = os.path.join(NDSB_DIR, 'resume_training_pl.sh')
SOLVER = os.path.join(NDSB_DIR, 'solver.prototxt')
NET = os.path.join(NDSB_DIR, 'train_val.prototxt')
CAFFE = '/afs/ee.cooper.edu/user/t/a/tam8/documents/caffe/build/tools/caffe'
MODELS_DIR = '/media/raid_arr/data/ndsb/models'
snapshot_prefix = 'alexmod_bruteaug_fold0_iter_'
MAX_ITER = 100000    # global max (not per step)
STEP = 100



def write_max_iter_to_solver(max_iter, f_path=SOLVER):
    with open(f_path, 'r') as f:
    # read a list of lines into data
        f_data = f.readlines()

    # Change the line with the PL loss weight
    line_n = 8    # The line number we're going to replace
    new_line = 'max_iter: ' + str(max_iter) + '\n'
    f_data[line_n] = new_line

    # and write everything back
    with open(f_path, 'w') as f:
        f.writelines(f_data)
    return f_path



snap_name = lambda n_iter: os.path.join(MODELS_DIR,
            snapshot_prefix + str(n_iter) + '.solverstate')

call_start = lambda sol=SOLVER: subprocess.call(
            [CAFFE, 'train', '--solver=' + sol])
call_resume = lambda snap, sol=SOLVER: subprocess.call(
            [CAFFE, 'train', '--solver=' + sol, '--snapshot=' + snap])



last_saved_iter = max({int(os.path.splitext(f)[0].rsplit('_', 1)[1]) 
                       for f in next(os.walk(MODELS_DIR))[2]}.union({0}))  

for ii in range(last_saved_iter, MAX_ITER+1, STEP):
    write_max_iter_to_solver(ii + STEP) # make caffe stop at the next step
    print 'ITER:\t', ii
    
    subprocess.call(['rm', '-rf', '/dev/shm/train0_aug_lvl'])
    subprocess.call(['cp', '-rf',
                     '/media/raid_arr/tmp/train0_aug_lvl',
                     '/dev/shm/train0_aug_lvl'])
    subprocess.call(['rm', '-rf', '/media/raid_arr/tmp/train0_aug_lvl'])
    # Start augmenting in another thread while caffe runs
    aug_thread = threading.Thread(target=my_io.create_aug_lvl)
    aug_thread.start()
    
    if ii == 0:
        print 'Starting new train'
        call_start()
        subprocess.call(['cp', '/tmp/caffe.INFO', 
                         '/tmp/my_caffe_log.txt'])
    else:
        call_resume(snap_name(ii))
        subprocess.call(['cat /tmp/caffe.INFO >> /tmp/my_caffe_log.txt'], shell=True)
print 'DONE'                        



