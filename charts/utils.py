import os
import pathlib
import matplotlib.pyplot as plt

def standard_fig_ax(x=1,y=1):
     return plt.subplots(x,y,figsize=(10,10))
    

def save_figures(filename, fig=None, dpi_list=[300, 600], formats=['jpg', 'svg']):
    pathlib.Path(f'output/{filename}').mkdir(parents=True,exist_ok=True)
    if fig is None:
        fig = plt.gcf()

    for dpi in dpi_list:
        for format in formats:
            fig.savefig(f'output/{filename}/{dpi}dpi.{format}', dpi=dpi, format=format)