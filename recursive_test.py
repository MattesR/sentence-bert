#!/usr/bin/env python

import sys
import os
import time

def just_recursive_printing(start, stop, over):
    print(sys.argv)
    for i in range(start,stop):
        print(f'ich bin eine {i}')
        time.sleep(2)
        if not i % over:
            print(f'huch eine {i}, na da mach ich dicht und springe zehn weiter')
            start = i+10
            arglist = ['filename', f'{start}', f'{stop}', f'{over}']
            sys.stdout.flush()
            os.execv(__file__, arglist)




if __name__ == "__main__":
    print(f'na das war ja mal was, ich mache weiter')
    args = sys.argv[1:]
    just_recursive_printing(int(args[0]), int(args[1]), int(args[2]))
