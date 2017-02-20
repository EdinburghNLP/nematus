import os
import sys
import random

import tempfile
from subprocess import call



def main(files, temporary=False):

    tf_os, tpath = tempfile.mkstemp()
    tf = open(tpath, 'w')

    fds = [open(ff) for ff in files]

    for l in fds[0]:
        lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
        print >>tf, "|||".join(lines)

    [ff.close() for ff in fds]
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)

    if temporary:
        fds = []
        for ff in files:
            path, filename = os.path.split(os.path.realpath(ff))
            fds.append(tempfile.TemporaryFile(prefix=filename+'.shuf', dir=path))
    else:
        fds = [open(ff+'.shuf','w') for ff in files]

    for l in lines:
        s = l.strip().split('|||')
        for ii, fd in enumerate(fds):
            print >>fd, s[ii]

    if temporary:
        [ff.seek(0) for ff in fds]
    else:
        [ff.close() for ff in fds]

    os.close(tf_os)
    os.remove(tpath)

    return fds

if __name__ == '__main__':
    main(sys.argv[1:])

    


