import os
import subprocess
# NDSB_DIR = '/afs/ee.cooper.edu/user/t/a/tam8/documents/ndsb2015'
NDSB_DIR = '/media/raid_arr/data/ndsb/config'
# TRAIN_SCRIPT = os.path.join(NDSB_DIR, 'train_pl.sh')
# RESUME_SCRIPT = os.path.join(NDSB_DIR, 'resume_training_pl.sh')
SOLVER = os.path.join(NDSB_DIR, 'solver_pl.prototxt')
NET = os.path.join(NDSB_DIR, 'train_val_pl.prototxt')
CAFFE = '/afs/ee.cooper.edu/user/t/a/tam8/documents/caffe/build/tools/caffe'
MODELS_DIR = '/media/raid_arr/data/ndsb/models'
snapshot_prefix = 'pl_iter_'
MAX_ITER = 100000    # global max (not per step)
STEP = 1000


def iter_to_epoch(ii):
    """
    Convert the iteration # to an approximate epoch
    ii: iteration
    """
    k = 30336.0/384    # Iterations per epoch (# training / batch size)
    return ii/k

def w_schedule(t):
    """
    Weight of the PL as a fn of iter
    t: epoch
    """
    wf = 3.0
    t1 = 100
    t2 = 600
    if t < t1:
        w = 0.0
    elif t > t2:
        w = wf
    else:
        w = wf*(float(t)-t1)/(t2-t1)
    return w

def write_weight_to_net(w, f_path=NET):
    """
    Could regex or import net,
    brute force much easier
    """
    with open(f_path, 'r') as f:
    # read a list of lines into data
        f_data = f.readlines()

    # Change the line with the PL loss weight
    line_n = 549    # The line number we're going to replace
    new_line = '  loss_weight: ' + str(w) + '\n'
    f_data[line_n] = new_line

    # and write everything back
    with open(f_path, 'w') as f:
        f.writelines(f_data)
    return f_path

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


last_saved_iter = max({int(os.path.splitext(f)[0].rsplit('_', 1)[1]) for f in next(os.walk(MODELS_DIR))[2]}.union({0}))    
for ii in range(last_saved_iter, MAX_ITER+1, STEP):
    e = iter_to_epoch(ii)
    w = w_schedule(e)
    write_weight_to_net(w)
    write_max_iter_to_solver(ii + STEP)
    print 'ITER:\t', ii, '\nWEIGHT:\t', w
    if ii == 0:
        call_start()
    else:
        call_resume(snap_name(ii))
            
print 'DONE'
