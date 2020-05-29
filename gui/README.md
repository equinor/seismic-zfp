# seismic-zfp desktop tool

The seismic-zfp desktop tool is a lightweight GUI built with tkinter using pyinstaller
to generate a standalone executable (can be run without a Python installation).

### Motivation
A setf-contained executable exposing important parts of the functionality of seismic-zfp 
(mainly the conversion between SEG-Y and SGZ formats) which can be readily distributed is 
desirable for making this format accessible to the wider geoscience community.

### Building
To build the GUI from this directory:
```
pip install requirements.txt
cp hook-seismic_zfp.py <your-site-packages>/PyInstaller/hooks/
pyinstaller seismic-zfp-desktop.py --onefile
```

### Usage
The GUI is designed to be self-explanatory.
