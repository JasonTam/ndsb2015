from collections import defaultdict
from itertools import takewhile
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
import os

curdir, _ = os.path.split(__file__)
DATA_PATH = '../data/chart_tabs.txt'
f_path_chart = os.path.join(curdir, DATA_PATH)


is_tab = '\t'.__eq__

def tree(): return defaultdict(tree)
def add(t, path):
    for node in path:
        t = t[node]
def dicts(t): return {k: dicts(t[k]) for k in t}

def build_tree(lines):
    lines = iter(lines)
    path = []
    ret = tree()
    for line in lines:
        entry = line.lstrip()
        indent = len(list(takewhile(is_tab, line)))
        path[indent:] = [entry]
        add(ret, path)
    return ret

def build_trace(lines):
    lines = iter(lines)
    path = []
    ret = []
    for line in lines:
        entry = line.lstrip()
        indent = len(list(takewhile(is_tab, line)))
        path[indent:] = [entry]
        if entry[0].lower() == entry[0]:
            ret.append(tuple(path[::-1]))
    return ret


with open(f_path_chart, 'r') as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

#t = build_tree(lines)
trace = build_trace(lines)

max_len = max([len(p) for p in trace])
trace_ext = []
for path in trace:
    while len(path) < max_len:
        path = (path[0],) + path
    trace_ext.append(path)

def print_width():
    for n in range(max_len):
        print n, len({p[n] for p in trace_ext})

trace_d = {p[0]: list(p) for p in trace_ext}
depth_le = {n: LabelEncoder().fit(sorted(list({p[n] for p in trace_ext}))) 
            for n in range(max_len)}

# takes in class label string and encodes the nth parent
encode_parent = lambda y_str, n: depth_le[n].transform(trace_d[y_str][n])






















