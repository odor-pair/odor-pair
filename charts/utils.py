import os
import matplotlib.pyplot as plt

def standard_fig_ax():
     return plt.subplots(figsize=(10,10))
    

def save_figures(filename, fig=None, dpi_list=[300, 600], formats=['jpg', 'svg']):
    os.mkdir(f'output/{filename}')
    if fig is None:
        fig = plt.gcf()

    for dpi in dpi_list:
        for format in formats:
            fig.savefig(f'output/{filename}/{dpi}dpi.{format}', dpi=dpi, format=format)
