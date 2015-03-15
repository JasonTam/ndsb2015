import os
import shutil as sh
import specialism as sp

def copy_coarse_to_fine(coarse_net, coarse_solver):
    # fine_net_paths = []
    fine_solver_paths = []
    # For each coarse class, prepare a separate net
    for c_lbl, f_lbls in sp.coarse_to_fine.items():

        #################### Copy over a new network definition
        fine_net_path = coarse_net.replace('coarse', 'fine' + str(c_lbl))
        sh.copy(coarse_net, fine_net_path)

        with open(fine_net_path, 'r') as f:
        # read a list of lines into data
            f_data = f.readlines()

        ####
        # Change the data source
        new_src_train = '\"/dev/shm/train0_aug_lvl' + '_' + str(c_lbl) + '\"'
        line_n = 9    # The line number we're going to replace
        new_line = '    source: ' + new_src_train + '\n'
        f_data[line_n] = new_line

        new_src_test = '\"/media/raid_arr/tmp/test0_norm_lmdb' + '_' + str(c_lbl) + '\"'
        line_n = 28    # The line number we're going to replace
        new_line = '    source: ' + new_src_test + '\n'
        f_data[line_n] = new_line

        # Change number of class prediction output
        line_n = 479    # The line number we're going to replace
        new_line = '    num_output: ' + str(len(f_lbls)) + '\n'
        f_data[line_n] = new_line

        ## Change the label to compare to
        #line_n = 497    # The line number we're going to replace
        #new_line = '  bottom: ' + '\"label2\"' + '\n'
        #f_data[line_n] = new_line

        # and write everything back
        with open(fine_net_path, 'w') as f:
            f.writelines(f_data)

        ##################### Copy over a new Solver definition
        fine_solver_path = coarse_solver.replace('coarse', 'fine' + str(c_lbl))
        sh.copy(coarse_solver, fine_solver_path)

        with open(fine_solver_path, 'r') as f:
        # read a list of lines into data
            f_data = f.readlines()

        ####
        # Change the net
        line_n = 0    # The line number we're going to replace
        new_line = 'net: ' + '\"' + fine_net_path + '\"' + '\n'
        f_data[line_n] = new_line

        line_n = 14    # The line number we're going to replace
        f_data[line_n] = f_data[line_n].replace('coarse', 'fine' + str(c_lbl))

        # and write everything back
        with open(fine_solver_path, 'w') as f:
            f.writelines(f_data)
        
        fine_solver_paths.append(fine_solver_path)
    return fine_solver_paths
