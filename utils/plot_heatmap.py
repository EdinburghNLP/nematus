import numpy
import matplotlib.pyplot as plt
import sys
import argparse

# input:
#  alignment matrix - numpy array
#  shape (target tokens + eos, number of hidden source states = source tokens +eos)
# one line correpsonds to one decoding step producing one target token
# each line has the attention model weights corresponding to that decoding step
# each float on a line is the attention model weight for a corresponding source state.
# plot: a head map of the alignment matrix
# x axis are the source tokens (alignment is to source hidden state that roughly corresponds to a source token)
# y azis are the target tokens

def plot_head_map(mma, target_labels, source_labels):
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

  # put the major ticks at the middle of each cell
  ax.set_xticks(numpy.arange(mma.shape[1])+0.5, minor=False)
  ax.set_yticks(numpy.arange(mma.shape[0])+0.5, minor=False)

  # want a more natural, table-like display
  ax.invert_yaxis()
  ax.xaxis.tick_top()


  # source words -> column labels
  ax.set_xticklabels(source_labels, minor=False)
  # target words -> row labels
  ax.set_yticklabels(target_labels, minor=False)

  plt.xticks(rotation=45)

  plt.show()


# column labels -> target words
# row labels -> source words


def read_alignment_matrix(f):
  header = f.readline().strip().split('|||')
  if header[0] == '':
    return None, None, None
  # number of tokens in source and translation +1 for eos
  src_count, trg_count = map(int,header[-1].split())
  # source words
  source_labels = header[3].decode('UTF-8').split()
  source_labels.append('</s>')
  # target words
  target_labels = header[1].decode('UTF-8').split()
  target_labels.append('</s>')
 
  mm = []
  for r in range(trg_count):
    alignment = map(float,f.readline().strip().split())
    mm.append(alignment)
  mma = numpy.array(mm)
  return mma, target_labels, source_labels


def read_plot_alignment_matrices(f, n):
  while(f):
    mma, target_labels, source_labels = read_alignment_matrix(f)
    if mma is None:
      return
    plot_head_map(mma, target_labels, source_labels)
    # empty line separating the matrices
    f.readline()


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        default='/Users/mnadejde/Documents/workspace/MTMA2016/models/wmt16_systems/en-de/test.alignment', metavar='PATH',
                        help="Input file (default: standard input)")

args = parser.parse_args()

read_plot_alignment_matrices(args.input,3)