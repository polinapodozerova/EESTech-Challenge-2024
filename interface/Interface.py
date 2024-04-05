from tkinter import *
from pandastable import Table, TableModel
import pandas as pd
# Lots of tutorials have from tkinter import *, but that is pretty much always a bad idea
from tkinter import ttk
import abc
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from data_preprocessing import preprocess_data
import pickle
import joblib
forecast_model = joblib.load(open('model.pkl', 'rb'))
anomaly_model = joblib.load(open('anomaly_model.pkl', 'rb'))
prob_history = []


class Menubar(Frame):
    """Builds a menu bar for the top of the main window"""
    def __init__(self, parent, *args, **kwargs):
        ''' Constructor'''
        Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_menubar()

    def on_exit(self):
        '''Exits program'''
        quit()

    def display_help(self):
        '''Displays help document'''
        pass

    def display_about(self):
        '''Displays info about program'''
        pass

    def init_menubar(self):
        self.menubar = Menu(self.root)
        self.menu_file = Menu(self.menubar) # Creates a "File" menu
        self.menu_file.add_command(label='Exit', command=self.on_exit) # Adds an option to the menu
        self.menubar.add_cascade(menu=self.menu_file, label='File') # Adds File menu to the bar. Can also be used to create submenus.

        self.menu_help = Menu(self.menubar) #Creates a "Help" menu
        self.menu_help.add_command(label='Help', command=self.display_help)
        self.menu_help.add_command(label='About', command=self.display_about)
        self.menubar.add_cascade(menu=self.menu_help, label='Help')

        self.root.config(menu=self.menubar)

class Window(Frame):
    """Abstract base class for a popup window"""
    __metaclass__ = abc.ABCMeta
    def __init__(self, parent, *args, **kwargs):
        ''' Constructor '''
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.resizable(width=False, height=False) # Disallows window resizing
        self.validate_notempty = (self.register(self.notEmpty), '%P') # Creates Tcl wrapper for python function. %P = new contents of field after the edit.
        self.init_gui()

    @abc.abstractmethod # Must be overwriten by subclasses
    def init_gui(self, df=None, i=None):
        '''Initiates GUI of any popup window'''
        pass

    @abc.abstractmethod
    def do_something(self):
        '''Does something that all popup windows need to do'''
        pass

    def notEmpty(self, P):
        '''Validates Entry fields to ensure they aren't empty'''
        if P.strip():
            valid = True
        else:
            print("Error: Field must not be empty.") # Prints to console
            valid = False
        return valid

    def close_win(self):
        '''Closes window'''
        self.parent.destroy()

class SomethingWindow(Window):
    """ New popup window """

    def __init__(self, parent, df):
        ''' Constructor '''
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.resizable(width=False, height=False)  # Disallows window resizing
        self.validate_notempty = (self.register(self.notEmpty),
                                  '%P')  # Creates Tcl wrapper for python function. %P = new contents of field after the edit.
        self.df = df
        self.init_gui(df)

    def init_gui(self, df=None, i=0):
        self.parent.title("New Window")
        self.parent.columnconfigure(1, weight=2)
        self.parent.rowconfigure(3, weight=1)

        # Create Widgets
        prob = i*5/100
        #prob = model.predict(df.iloc[i])

        self.time = ttk.Label(self.parent, text=f"Дата и время: {str(df.iloc[i][0])}", width=60)
        self.val1 = ttk.Label(self.parent, text=f"Полож.пед.акселер.,%: {str(df.iloc[i][1])}", relief="sunken", width=60)
        self.val2 = ttk.Label(self.parent, text=f"Давл.масла двиг.,кПа: {str(df.iloc[i][2])}", relief="sunken", width=60)
        self.val3 = ttk.Label(self.parent, text=f"КПП. Давление масла в системе смазки: {str(df.iloc[i][8])}", relief="sunken", width=60)
        self.val4 = ttk.Label(self.parent, text=f"Обор.двиг.,об/мин: {str(df.iloc[i][4])}", relief="sunken", width=60)
        self.val5 = ttk.Label(self.parent, text=f"Значение счетчика моточасов, час:мин: {str(df.iloc[i][5])}", relief="sunken", width=60)
        self.val6 = ttk.Label(self.parent, text=f": {str(df.iloc[i][6])}", relief="sunken", width=60)
        self.probability = ttk.Label(self.parent, text=f"Вероятность поломки: {prob}", relief="sunken", width=60)
        self.pb1 = ttk.Progressbar(self.parent, orient=HORIZONTAL, length=100)
        self.pb1['value'] = prob*100
        #self.f = ttk.Frame(self.parent, relief="sunken", height = 80, width = )

        #self.table = pt = Table(self.f, dataframe=df.iloc[i:i+1],
        #                       showtoolbar=True, showstatusbar=True)
        i += 1
        #pt.show()

        # Layout
        self.time.grid(row=1, column=1, columnspan=2, sticky='nsew')
        #self.f.grid(row=3, column=0, columnspan=2, sticky='nsew')
        self.val1.grid(row=3, column=1, columnspan=2, sticky='nsew')
        self.val2.grid(row=3, column=3, columnspan=2, sticky='nsew')
        self.val3.grid(row=5, column=1, columnspan=2, sticky='nsew')
        self.val4.grid(row=5, column=3, columnspan=2, sticky='nsew')
        self.val5.grid(row=6, column=1, columnspan=2, sticky='nsew')
        self.val6.grid(row=6, column=3, columnspan=2, sticky='nsew')
        self.pb1.grid(row=7, column=3, columnspan=2)
        self.probability.grid(row=7, column=1, columnspan=2, sticky='nsew')


        # Padding
        for child in self.parent.winfo_children():
            child.grid_configure(padx=10, pady=5)

        self.after(1000, self.change_time, df, i)

    def change_time(self, df, i):
        self.time.config(text=f"Время и дата: {str(df.iloc[i][0])}")
        #self.table = pt = Table(self.f, dataframe=df.iloc[i:i + 1],
        #                        showtoolbar=True, showstatusbar=True)
        i += 1
        #self.table.show()
        anomaly = anomaly_model.predict_proba(((pd.DataFrame(df.iloc[i])).T).drop(columns='Дата и время'))[:,1][0]
        prob = forecast_model.predict_proba(((pd.DataFrame(df.iloc[i])).T).drop(columns='Дата и время'))[:,1][0]
        prob_history.append(prob)
        #prob = model.predict(df.iloc[i])
        self.val1.config(text=f"Полож.пед.акселер.,%: {str(df.iloc[i][1])}", relief="sunken")
        self.val2.config(text=f"Давл.масла двиг.,кПа: {str(df.iloc[i][2])}", relief="sunken")
        self.val3.config(text=f"КПП. Давление масла в системе смазки: {str(df.iloc[i][8])}", relief="sunken")
        self.val4.config(text=f"Обор.двиг.,об/мин: {str(df.iloc[i][4])}", relief="sunken")
        self.val5.config(text=f"Значение счетчика моточасов, час:мин: {str(df.iloc[i][5])}", relief="sunken")
        self.val6.config(text=f": {str(df.iloc[i][6])}", relief="sunken")
        self.probability.config(text=f"Вероятность поломки: {prob}", relief="sunken")
        if anomaly == 0:
            self.pb1['value'] = round(prob*100)
        self.after(1000, self.change_time, df, i)


class GUI(Frame):
    """Main GUI class"""
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def openwindow(self, df):
        self.new_win = Toplevel(self.root) # Set parent
        SomethingWindow(self.new_win, df)

    def load_file(self):
        fname = askopenfilename(filetypes=(("All files", "*.*"),
                                           ("CSV files", "*.csv")))
        if fname:
            try:
                df = pd.read_csv(fname, sep=';', low_memory="False")
                df = preprocess_data(df)
                self.openwindow(df)
            except:                     # <- naked except is a bad idea
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return

    def init_gui(self):
        self.root.title('Test GUI')
        self.root.geometry("600x400")
        self.grid(column=0, row=0, sticky='nsew')
        self.grid_columnconfigure(0, weight=1) # Allows column to stretch upon resizing
        self.grid_rowconfigure(0, weight=1) # Same with row
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.option_add('*tearOff', 'FALSE') # Disables ability to tear menu bar into own window
        
        # Menu Bar
        self.menubar = Menubar(self.root)
        
        # Create Widgets
        self.btn = ttk.Button(self, text='Загрузить файл данных с трактора', command=self.load_file)

        # Layout using grid
        self.btn.grid(row=0, column=0, sticky='ew')

        # Padding
        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

if __name__ == '__main__':
    root = Tk()
    GUI(root)
    root.mainloop()

