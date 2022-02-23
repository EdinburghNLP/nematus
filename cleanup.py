from os import listdir
from os.path import isfile, join

def cleanup():
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

if __name__ == '__main__':
