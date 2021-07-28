import tkinter.filedialog as filedialog
import tkinter as tk
import seismic_zfp
from seismic_zfp.conversion import SgzReader, SegyConverter, SgzConverter
from datetime import timedelta
import time
import os

HEIGHT = 500
WIDTH = 700

BLACK = '#000000'
WHITE = '#FFFFFF'
ENERGY_8 = '#FFECF0'
ENERGY_30 = '#FFB7C6'
ENERGY_50 = '#FF88A1'
ENERGY_70 = '#FF5A7B'
ENERGY_RED = '#FF1243'

filetype_options_sgy = (('SEG-Y files','*.sgy *.segy'),
                        ('SGZ','*.sgz'),
                        ('all files','*.*'))

filetype_options_sgz = (('SGZ','*.sgz'),
                        ('SEG-Y files','*.sgy *.segy'),
                        ('all files','*.*'))

def get_in_filename():
    if compress.get():
        filename = filedialog.askopenfilename(initialdir='/',
                                              title='Select file',
                                              filetypes=filetype_options_sgy)
    else:
        filename = filedialog.askopenfilename(initialdir='/',
                                              title='Select file',
                                              filetypes=filetype_options_sgz)
    infile.set(filename)
    show_choices()

def get_out_filename():
    if compress.get():
        filename = filedialog.asksaveasfilename(initialdir = '/',
                                                title = 'Select file',
                                                filetypes=filetype_options_sgz)
    else:
        filename = filedialog.asksaveasfilename(initialdir = '/',
                                                title = 'Select file',
                                                filetypes=filetype_options_sgy)
    outfile.set(filename)
    show_choices()

def update_messages(new_line):
    message.set(message.get() + '\n' + new_line)

def get_license():
    license_window = tk.Toplevel(root)
    license_window.minsize(WIDTH, 80)
    canvas = tk.Canvas(license_window, width=WIDTH, height=80, bg=ENERGY_8)
    canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
    label = tk.Label(license_window, text='seismic-zfp license is: GNU LGPL v3\n\nZFP compression license is: BSD 3-clause', justify='left', anchor='w', font=('bold', 18), fg=WHITE, bg=ENERGY_RED, bd=5)
    label.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)


def update_action():
    if compress.get():
        run_button.configure(text="Compress")
    else:
        run_button.configure(text="Decompress")

def get_infile_size_gb():
    infile_size_bytes = os.path.getsize(infile.get())
    infile_size_gb = infile_size_bytes/(1024*1024*1024)
    update_messages('Input file size: {:8.2f}GB'.format(infile_size_gb))
    return infile_size_gb

def show_choices():
    try:
        filename, file_extension = os.path.splitext(infile.get())
        if file_extension.upper() in [".SGY", ".SEGY"]:
            compress.set(True)
            if adv_layout.get():
                bitrate.set("2")
                outfile.set(infile.get().replace('.sgy', '_adv.sgz'))
            else:
                outfile.set(infile.get().replace('.segy', '.sgy').replace('.sgy', '_{}.sgz'.format(bitrate.get())))
            update_action()
            infile_size_gb = get_infile_size_gb()
            outfile_size_gb = ((float(bitrate.get())/32.0)*infile_size_gb)
            update_messages('Expected output file size: {:8.2f}GB'.format(outfile_size_gb))

        elif file_extension.upper() == ".SGZ":
            compress.set(False)
            update_action()
            with SgzReader(infile.get()) as reader:
                bitrate.set(str(reader.rate))
                outfile_size_gb = ((reader.n_samples * 4 + 240) * reader.n_xlines * reader.n_ilines)/(1024*1024*1024)
                blockshape = reader.blockshape
            if reader.blockshape == (64, 64, 4):
                update_messages('Error: Cannot decmopress "adv" SGZ to SEG-Y')
            else:
                get_infile_size_gb()
                update_messages('Expected output file size: {:8.2f}GB'.format(outfile_size_gb))
                outfile.set(infile.get().replace('.sgz', '.sgy'.format(bitrate.get())))
        else:
            update_messages('Unrecognized extension: use ".sgy", ".segy", ".sgz"')

    except FileNotFoundError:
        update_messages('File not found: ' + infile.get())
    
    finally:
        update_action()

def run_action():
    try:
        t0 = time.time()
        if compress.get():
            update_messages('Running compression...')
            if adv_layout.get():
                with SegyConverter(infile.get()) as converter:
                        converter.run(outfile.get(), 
                                      bits_per_voxel=2,
                                      blockshape=(64, 64, 4),
                                      reduce_iops=True)
            else:
                with SegyConverter(infile.get()) as converter:
                        converter.run(outfile.get(),
                                      bits_per_voxel=bitrate.get(), 
                                      reduce_iops=True)
        else:
            update_messages('Running decompression...')
            with SgzConverter(infile.get()) as converter:
                converter.convert_to_segy(outfile.get())
        tt = time.time() - t0
        
        outfile_size_bytes = os.path.getsize(outfile.get())
        outfile_size_gb = (outfile_size_bytes)/(1024*1024*1024)

        update_messages('...complete.')
        update_messages('Actual output file size: {:8.2f}GB'.format(outfile_size_gb))
        update_messages('Time taken: {}'.format(str(timedelta(seconds=tt))))
        
    except FileNotFoundError:
        update_messages('File not found: ' + infile.get())

root = tk.Tk()
root.minsize(WIDTH, HEIGHT)

bitrate = tk.StringVar()
bitrate.set("4")

infile = tk.StringVar()
infile.set('Input File')

outfile = tk.StringVar()
outfile.set('Output File')

message = tk.StringVar()
message.set('')

compress = tk.BooleanVar()
compress.set(True)

adv_layout = tk.BooleanVar()
adv_layout.set(False)

bitrates = [('4:1',"8"), ('8:1',"4"), ('16:1',"2"), ('32:1',"1"), ('64:1',"0.5")]

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=ENERGY_8)
canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

label = tk.Label(root, text='   seismic-zfp Desktop Tool', anchor='w', font=('bold', 24), fg=WHITE, bg=ENERGY_RED, bd=5)
label.place(relx=0.05, y=20, relwidth=0.9, height=40)

license_button = tk.Button(label, text='License', command=get_license)
license_button.place(relx=0.85, rely=0, relheight=1, width=80)


upper_frame = tk.Frame(root, bg=ENERGY_70, bd=5)
upper_frame.place(relx=0.05, y=80, relwidth=0.9, height=40)

upper_entry = tk.Entry(upper_frame, font=30, textvariable=infile)
upper_entry.place(relx=0.025, relwidth=0.8, relheight=1)

upper_button = tk.Button(upper_frame, text='Select', command=get_in_filename)
upper_button.place(relx=0.85, rely=0, relheight=1, width=80)


lower_frame = tk.Frame(root, bg=ENERGY_50, bd=5)
lower_frame.place(relx=0.05, y=140, relwidth=0.9, height=40)

lower_entry = tk.Entry(lower_frame, font=30, textvariable=outfile)
lower_entry.place(relx=0.025, relwidth=0.8, relheight=1)

lower_button = tk.Button(lower_frame, text='Select', command=get_out_filename)
lower_button.place(relx=0.85, rely=0, relheight=1, width=80)


run_frame = tk.Frame(root, bg=ENERGY_30, bd=5)
run_frame.place(relx=0.05, y=200, relwidth=0.9, relheight=0.55)

bitrate_selector = tk.Frame(run_frame, bg=ENERGY_30, bd=5)
bitrate_selector.place(relx=0, rely=0, relwidth=0.15, relheight=1)
tk.Label(bitrate_selector, text='Ratio', font=20, bg=ENERGY_30).pack(anchor='w')

for (text, integer) in bitrates:
    tk.Radiobutton(bitrate_selector,
                   bg=ENERGY_30,
                   command=show_choices, 
                   variable=bitrate, 
                   text=text, 
                   font=18,
                   value=integer).pack(anchor='w')

message_frame = tk.Frame(run_frame, bg=ENERGY_30)
message_frame.place(relx=0.15, rely=0.05, relwidth=0.85, relheight=0.7)

message_label = tk.Label(message_frame, font=("Fixedsys", 10), bg=BLACK, fg=WHITE, anchor='sw', justify='left', textvariable=message)
message_label.place(relx=0, rely=0, relwidth=1, relheight=1)

adv_checkbox = tk.Checkbutton(run_frame, text="Advanced layout", font=20, bg=ENERGY_30, variable=adv_layout, command=show_choices)
adv_checkbox.place(relx=0.25, rely=0.8, relwidth=0.4, height=40)

run_button = tk.Button(run_frame, font=24, text="Compress", command=run_action)
run_button.place(relx=0.7, rely=0.8, relwidth=0.3, height=40)

root.mainloop()