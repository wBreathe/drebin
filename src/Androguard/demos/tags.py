#!/usr/bin/env python

import sys

PATH_INSTALL = "./"
sys.path.append(PATH_INSTALL)

from Androguard.androguard.core.bytecodes import dvm
from Androguard.androguard.core.bytecodes import apk
from Androguard.androguard.core.analysis import analysis
from Androguard.androguard.core import androconf


import hashlib

TEST = "examples/android/TestsAndroguard/bin/TestsAndroguard.apk"

androconf.set_debug()

a = apk.APK( TEST )
vm = dvm.DalvikVMFormat( a.get_dex() )
vmx = analysis.VMAnalysis( vm )

for i in vmx.get_methods() :
    i.create_tags()

    tags = i.get_tags()
    if not tags.empty() :
        print(tags)
