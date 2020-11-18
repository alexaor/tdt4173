import pathlib
import os
from colorama import Fore, Style
import shutil

# Make a results directory if it does not yet exist
result_dir = pathlib.Path('results')
result_dir.mkdir(exist_ok=True, parents=True)

# Make a plots directory inside the result directory if it does not yet exist
plots_dir = pathlib.Path(os.path.join(result_dir, 'plots'))
plots_dir.mkdir(exist_ok=True, parents=True)


"""
:param filename:    str, the name of the file, need to have file extension '.txt'
:param text:        str, the content that shall be written to the file

:return :           str, if successful the filepath, if not -1

Makes (or overwrite) a file with the given content, the file will be saved in the directory 'results'. If the file
extension is wrong, it will print a warning and return the string '-1'.
"""
def save_text(filename, text):
    if filename.endswith('.txt'):
        filepath = os.path.join(result_dir, filename)
        f = open(filepath, "w")
        f.write(text)
        f.close()
        return filepath
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{filename.split(".")[-1]}" \t--> should be ".txt"')
        print(Style.RESET_ALL)
        return '-1'


"""
:param dirname:     str, name of the new directory in the plots directory

Checks if there is a directory in 'results/plots' with the same name, if it is, the directory's content will be deleted
to ensure that there are no problems with overwriting and only the wanted plots will exist in this folder. Then it will
make a new folder with the path: 'results/plots/<dirname>'.
"""
def make_plot_dir(dirname):
    plot_dir_path = pathlib.Path(os.path.join(plots_dir, dirname))
    # Make sure that the folder is empty, so there will be no problems with overwriting
    if os.path.isdir(plot_dir_path):
        shutil.rmtree(plot_dir_path)
    plot_dir_path.mkdir(exist_ok=True, parents=True)


"""
:param plot:                pyplot, the plot figure which shall be saved
:param plotname:            str, the name of what the plot will be saved as
:param dirname:             str, name of the directory for the plots to be saved in, if empty saved in 'results/plots'

:return plot_dir_path:      str, the path where the plots have been saved.

Saves the plot as the given plotname, if dirname is specified the path for the saved plot will be
'results/<plotname>/<dirname>'. If 'dirname' is specified but not exist, the program will exit with an error message.
"""
def save_plot(plot, plotname, dirname=""):
    plot_dir_path = pathlib.Path(os.path.join(plots_dir, dirname))
    if not os.path.isdir(plot_dir_path):
        print(Fore.RED + f'ERROR: Could not find directory: {plot_dir_path}')
        print(Style.RESET_ALL)
        exit(1)
    plot_path = os.path.join(plot_dir_path, plotname)
    plot.savefig(plot_path)
    return plot_path





