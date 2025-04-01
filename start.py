#!/usr/bin/env python3

import subprocess
import os

# Activate the virtual environment if it exists
venv_path = os.path.join(os.getcwd(), 'venv', 'bin', 'activate')
if os.path.exists(venv_path):
    print("Activating virtual environment...")
    activate_this_file = venv_path
    exec(open(activate_this_file).read(), {'__file__': activate_this_file})

# Launch the main program
print("Launching wd14_web_gui.py...")
subprocess.run(['python', 'wd14_web_gui.py'])