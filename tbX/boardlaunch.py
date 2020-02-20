# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:42:18 2019

@author: Khashayar (ERNEST) Namdar
"""

import LTRI_Funcs as LTRIf
from tensorboard import program
import webbrowser

print("Run this file in a new console so that you can continue your work.\n")
print("Select the log file directory")
lgdir = LTRIf.get_dirname()
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', lgdir, '--host', '127.0.0.1'])
url = tb.launch()
webbrowser.open_new(url)
