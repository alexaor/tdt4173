import pathlib
import shutil
import os

from colorama import Fore, Style

from configs.project_settings import ROOT_RESULTS_DIR_PATH

# Make a results directory if it does not yet exist
RESULT_DIR = pathlib.Path(ROOT_RESULTS_DIR_PATH)
RESULT_DIR.mkdir(exist_ok=True, parents=True)

# Make a plots directory inside the result directory if it does not yet exist
PLOTS_DIR = pathlib.Path(os.path.join(RESULT_DIR, 'plots'))
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def save_text(filename, text):
    """
    Makes (or overwrite) a file with the given content, the file will be saved in the directory 'results'. If the file
    extension is wrong, it will print a warning and return the string '-1'.

    Parameters
    ----------

    filename : str
        The name of the file, need to have file extension '.txt'
    text : str
        The content that shall be written to the file

    Returns
    -------
    filepath : str
        If successful the filepath, if not -1
    """
    
    if filename.endswith('.txt'):
        filepath = os.path.join(RESULT_DIR, filename)
        f = open(filepath, "w")
        f.write(text)
        f.close()
        return filepath
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{filename.split(".")[-1]}" \t--> should be ".txt"')
        print(Style.RESET_ALL)
        return '-1'


def make_plot_dir(dirname) -> None:
    """
    Generates folder to store plots

    Checks if there is a directory in 'results/plots' with the same name, if it is, the directory's content will be deleted
    to ensure that there are no problems with overwriting and only the wanted plots will exist in this folder. Then it will
    make a new folder with the path: 'results/plots/<dirname>'.

    Parameters
    ----------

    dirname : str
        Name of the new directory in the plots directory
    """

    plot_dir_path = pathlib.Path(os.path.join(PLOTS_DIR, dirname))
    # Make sure that the folder is empty, so there will be no problems with overwriting
    if os.path.isdir(plot_dir_path):
        shutil.rmtree(plot_dir_path)
    plot_dir_path.mkdir(exist_ok=True, parents=True)


def save_plot(plot, plotname, dirname=""):
    """
    Saves a plot to the specified directory

    Saves the plot as the given plotname, if dirname is specified the path for the saved plot will be
    'results/<plotname>/<dirname>'. If 'dirname' is specified but not exist, the program will exit with an error message

    Parameters
    ----------
    plot : pyplot
        The plot figure which shall be saved
    plotname : str
        The name of what the plot will be saved as
    dirname : str, optional
        Name of the directory for the plots to be saved in, if empty saved in 'results/plots'

    Returns
    -------
    plot_dir_path : str
        The path where the plots have been saved.
    """

    plot_dir_path = pathlib.Path(os.path.join(PLOTS_DIR, dirname))
    if not os.path.isdir(plot_dir_path):
        print(Fore.RED + f'ERROR: Could not find directory: {plot_dir_path}')
        print(Style.RESET_ALL)
        exit(1)
    plot_path = os.path.join(plot_dir_path, plotname)
    plot.savefig(plot_path)
    return plot_path
