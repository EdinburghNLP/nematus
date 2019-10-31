#!/usr/bin/env python3

import math
import os
import random
import sys
import tempfile


# TODO Make CHUNK_SIZE user configurable?
CHUNK_SIZE = 10000000  # Number of lines.

def jointly_shuffle_files(files, temporary=False):
    """Randomly shuffle the given files, applying the same permutation to each.

    Since the same permutation is applied to all input files, they must
    contain the same number of input lines.

    If 'temporary' is True then the shuffled files are written to temporary
    files. Otherwise, the shuffled files are written to files with the same
    paths as the originals, but with the added suffix '.shuf'.

    In addition to shuffling the files, any leading or trailing whitespace is
    removed from each line.

    In order to handle large files, the input files are not read into memory
    in full, but instead are read in chunks of size CHUNK_SIZE.

    Args:
        files: a list of strings specifying the paths of the input files.
        temporary: a Boolean (see description above).

    Returns:
        A list containing a file object for each shuffled file, in the same
        order as the input files. Each file object is open and positioned at
        the start of the file.
    """

    # Determine the number of lines (should be the same for all files).
    total_lines = 0
    for _ in open(files[0]):
        total_lines += 1

    # Randomly permute the list of line numbers.
    perm = list(range(total_lines))
    random.shuffle(perm)

    # Convert the list of line numbers to a list of chunk indices and offsets.
    ordering = [(i // CHUNK_SIZE, i % CHUNK_SIZE) for i in perm]

    # Sort each file according to the generated ordering.
    return [_sort_file(path, ordering, temporary) for path in files]


def _sort_file(path, ordering, temporary):

    # Open a temporary file for each chunk.

    num_chunks = math.ceil(len(ordering) / CHUNK_SIZE)
    dirname, filename = os.path.split(os.path.realpath(path))
    chunk_files = [tempfile.TemporaryFile(prefix=filename+'.chunk'+str(i),
                                          dir=dirname, mode='w+',
                                          encoding="UTF-8")
                   for i in range(num_chunks)]

    # Read one chunk at a time from path and write the lines to the temporary
    # files in the order specified by ordering.

    def _write_chunk_in_order(chunk, chunk_num, out_file):
        for i, j in ordering:
            if i == chunk_num:
                out_file.write(chunk[j] + '\n')

    chunk = []
    chunk_num = 0
    for i, line in enumerate(open(path)):
        if i > 0 and (i % CHUNK_SIZE) == 0:
            _write_chunk_in_order(chunk, chunk_num, chunk_files[chunk_num])
            chunk = []
            chunk_num += 1
        chunk.append(line.strip())
    if chunk:
        _write_chunk_in_order(chunk, chunk_num, chunk_files[chunk_num])

    # Open the output file.
    if temporary:
        out_file = tempfile.TemporaryFile(prefix=filename+'.shuf', dir=dirname,
                                          mode='w+', encoding='UTF-8')
    else:
        out_file = open(path+'.shuf', mode='w', encoding='UTF-8')

    # Seek to the start of the chunk files.
    for chunk_file in chunk_files:
        chunk_file.seek(0)

    # Write the output.
    for i, _ in ordering:
        line = chunk_files[i].readline()
        out_file.write(line)

    # Seek to the start so that the file object is ready for reading.
    out_file.seek(0)

    return out_file


if __name__ == '__main__':
    jointly_shuffle_files(sys.argv[1:])
