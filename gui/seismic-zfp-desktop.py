import tkinter.filedialog as filedialog
import tkinter as tk
from seismic_zfp.conversion import SegyConverter
from datetime import timedelta
import time
import os

HEIGHT = 450
WIDTH = 600

WHITE = '#FFFFFF'
ENERGY_8 = '#FFECF0'
ENERGY_30 = '#FFB7C6'
ENERGY_50 = '#FF88A1'
ENERGY_70 = '#FF5A7B'
ENERGY_RED = '#FF1243'

def get_in_filename():
    filename = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('SEG-Y files','*.sgy *.segy'),('all files','*.*')))
    infile.set(filename)
    show_choices()

def get_out_filename():
    filename = filedialog.asksaveasfilename(initialdir = '/',title = 'Select file',filetypes = (('SGZ','*.sgz'),('all files','*.*')))
    outfile.set(filename)
    show_choices()

def show_choices():
    try:
        infile_size_bytes = os.path.getsize(infile.get())
        infile_size_gb = infile_size_bytes/(1024*1024*1024)
        outfile_size_gb = ((float(bitrate.get())/32.0)*infile_size_gb)
        message.set('Input file size: {:8.2f}GB'.format(infile_size_gb))
        message.set(message.get() + '\nExpected output file size: {:8.2f}GB'.format(outfile_size_gb))
        outfile.set(infile.get().replace('.sgy', '_{}.sgz'.format(bitrate.get())))
    except FileNotFoundError:
        message.set('File not found')

def run_compression():
    try:
        t0 = time.time()
        with SegyConverter(infile.get()) as converter:
                converter.run(outfile.get(), bits_per_voxel=bitrate.get(), method='Stream')
        tt = time.time() - t0
        
        outfile_size_bytes = os.path.getsize(outfile.get())
        outfile_size_gb = (outfile_size_bytes)/(1024*1024*1024)

        message.set(message.get() + '\nActual output file size: {:8.2f}GB'.format(outfile_size_gb))
        message.set(message.get() + '\nTime taken: {}'.format(str(timedelta(seconds=tt))))
        
    except FileNotFoundError:
        message.set('File not found')

root = tk.Tk()
root.minsize(WIDTH, HEIGHT)

bitrate = tk.IntVar()
bitrate.set(4)

infile = tk.StringVar()
infile.set('Input File')

outfile = tk.StringVar()
outfile.set('Output File')

message = tk.StringVar()
message.set('')

bitrates = [('8',8), ('4',4), ('2',2), ('1',1)]

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=ENERGY_8)
canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

label = tk.Label(root, text='SEG-Y to SGZ Desktop', font=('bold', 28), fg=WHITE, bg=ENERGY_RED)
label.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.1)

upper_frame = tk.Frame(root, bg=ENERGY_70, bd=5)
upper_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.1, anchor='nw')

upper_entry = tk.Entry(upper_frame, font=30, textvariable=infile)
upper_entry.place(relx=0.025, relwidth=0.7, relheight=1)

upper_button = tk.Button(upper_frame, text='Select', command=get_in_filename)
upper_button.place(relx=0.8, relheight=1, relwidth=0.2)


lower_frame = tk.Frame(root, bg=ENERGY_50, bd=5)
lower_frame.place(relx=0.05, rely=0.35, relwidth=0.9, relheight=0.1, anchor='nw')

lower_entry = tk.Entry(lower_frame, font=30, textvariable=outfile)
lower_entry.place(relx=0.025, relwidth=0.7, relheight=1)

lower_button = tk.Button(lower_frame, text='Select', command=get_out_filename)
lower_button.place(relx=0.8, relheight=1, relwidth=0.2)


bitrate_selector = tk.Frame(root, bg=ENERGY_30, bd=5)
bitrate_selector.place(relx=0.05, rely=0.5, relwidth=0.1, relheight=0.4)
tk.Label(bitrate_selector, text='Bitrate', font=20, bg=ENERGY_30).pack(anchor='w')

for (text, integer) in bitrates:
    tk.Radiobutton(bitrate_selector,
                   bg=ENERGY_30,
                   command=show_choices, 
                   variable=bitrate, 
                   text=text, 
                   font=18,
                   value=integer).pack(anchor='w')

run_frame = tk.Frame(root, bg=ENERGY_30, bd=5)
run_frame.place(relx=0.15, rely=0.5, relwidth=0.8, relheight=0.4)

message_frame = tk.Frame(run_frame, bg=ENERGY_30)
message_frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.5)

message_label = tk.Label(message_frame, font=20, bg=ENERGY_30, textvariable=message)
message_label.place(relx=0, rely=0, relwidth=1, relheight=1)
                   
run_button = tk.Button(run_frame, font=24, text='Run', command=run_compression)
run_button.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.2)

root.mainloop()