import pickle
import os

curdir, _ = os.path.split(__file__)
f_path_le = os.path.join(curdir, './le.p')
f_path_scaler = os.path.join(curdir, './scaler.p')

le = pickle.load(open(f_path_le, 'rb'))

scaler = pickle.load(open(f_path_scaler, 'rb'))
