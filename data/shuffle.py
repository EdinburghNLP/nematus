import os
import sys
import random

import tempfile
from subprocess import call


def main(files, temporary=False):

    fds = [open(ff, encoding="UTF-8") for ff in files]

    lines = []
    for l in fds[0]:
        line = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
        lines.append(line)

    [ff.close() for ff in fds]

    random.shuffle(lines)

    if temporary:
        fds = []
        for ff in files:
            path, filename = os.path.split(os.path.realpath(ff))
            fd = tempfile.TemporaryFile(prefix=filename+'.shuf',
                                        dir=path,
                                        mode='w+',
                                        encoding="UTF-8")
            fds.append(fd)
    else:
        fds = [open(ff+'.shuf', mode='w', encoding="UTF-8") for ff in files]

    for l in lines:
        for ii, fd in enumerate(fds):
            print(l[ii], file=fd)

    if temporary:
        [ff.seek(0) for ff in fds]
    else:
        [ff.close() for ff in fds]

    return fds

if __name__ == '__main__':
    main(sys.argv[1:])

    


