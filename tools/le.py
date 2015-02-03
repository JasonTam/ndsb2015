import pickle
import os

curdir, _ = os.path.split(__file__)
f_path = os.path.join(curdir, './le.p')

le = pickle.load(open(f_path, 'rb'))
