#!/usr/bin/env python

import sys

PATH_INSTALL = "./"
sys.path.append(PATH_INSTALL)

from Androguard.androguard.core.bytecodes import dvm
from Androguard.androguard.core.analysis import analysis

TEST = "examples/android/TestsAndroguard/bin/classes.dex"

j = dvm.DalvikVMFormat( open(TEST).read() )
x = analysis.VMAnalysis( j )
j.set_vmanalysis( x )

# SHOW CLASSES (verbose and pretty)
j.pretty_show()

# SHOW METHODS
for i in j.get_methods() :
    i.pretty_show( )
