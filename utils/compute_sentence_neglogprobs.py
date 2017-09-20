import sys
import numpy

def main():
    for i, line in enumerate(sys.stdin):
        if (i % 2) == 0:
            continue
        probs = numpy.array(map(float, line.split()))
        neglogprob = -numpy.log(probs).sum()
        print neglogprob

if __name__ == '__main__':
    main()


