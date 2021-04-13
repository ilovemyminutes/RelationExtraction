import pickle

def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

def set_korfont(size: int=9):
    fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
    font = fm.FontProperties(fname=fontpath, size=9)