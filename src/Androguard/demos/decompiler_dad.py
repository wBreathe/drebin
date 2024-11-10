#!/usr/bin/env python

import sys

PATH_INSTALL = "./"
sys.path.append(PATH_INSTALL)

from Androguard.androguard.core.bytecodes import dvm
from Androguard.androguard.core.analysis import analysis
from Androguard.androguard.decompiler.dad import decompile

TEST = 'examples/android/TestsAndroguard/bin/classes.dex'

vm = dvm.DalvikVMFormat(open(TEST, "r").read())
vmx = analysis.VMAnalysis(vm)

# CFG
for method in vm.get_methods():
    mx = vmx.get_method(method)

    if method.get_code() == None:
      continue

    print(method.get_class_name(), method.get_name(), method.get_descriptor())

    ms = decompile.DvMethod(mx)
    ms.process()

    print(ms.get_source())
