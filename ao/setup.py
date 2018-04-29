#!/usr/bin/env python

import sys
import os

def print_y(s):
	print('\033[1;33m' + s + '\033[m')
	return

def print_bw(s):
	print('\033[1;1m' + s + '\033[m')
	return

def errorcheck_sys(error):
	if error:
		print_y("Error occurs. Please run setup.py first to compile the binary files.")
		sys.exit()
	return

def errorcheck_prog(error):
	if error:
		print_y("Binary error occurs.")
		sys.exit()
	return

print_bw("Compiling AO_CWS code ...")

if not os.path.exists("build"):
	errorcheck_sys(os.mkdir("build"))
errorcheck_sys(os.chdir("build"))
errorcheck_sys(os.system("cmake .."))
errorcheck_sys(os.system("make -j8"))
errorcheck_sys(os.system("make install"))

print_bw("You are done. AO_CWS CUDA code compilation is finished. Program exits.")
