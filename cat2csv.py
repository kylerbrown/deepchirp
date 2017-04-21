import numpy as np
import bark
from scipy import stats

default_min_syl = 30
default_min_silent = 15
def first_pass(cats, decoder, time_interval):
    '''cats:  a vector of length N, where N is the number of targets
        decoder: a map from values in cats to a syllable name
        time_interval: the time between to rows in cats (inverse of sampling rate)'''
    starts = []
    stops = []
    names = []
    i = 0 
    while i < len(cats):
        if cats[i] > 0:
            # find end
            i_end = i
            while cats[i_end] > 0:
                i_end += 1
            m = stats.mode(cats[i: i_end-1])[0]
            if len(m) == 0:
                syl_cat = cats[i]
            else:
                syl_cat = int(m)
            names.append(decoder[syl_cat])
            starts.append(i * time_interval)
            stops.append(i_end * time_interval)
            i = i_end
        else:
            i += 1
    return starts, stops, names

def second_pass(starts, stops, names, min_silent):
    ' If two syllables are within min_silent, join them'
    i = 1
    while i < len(starts):
        if starts[i] - stops[i - 1] <= min_silent and names[i] == names[i - 1]:
            stops[i - 1] = stops[i]
            del starts[i]
            del stops[i]
            del names[i]
        else:
            i += 1


def third_pass(starts, stops, names, min_syl):
    ' If a syllable is too short, remove'
    i = 0
    while i < len(starts):
        if stops[i] - starts[i] <= min_syl:
            del starts[i]
            del stops[i]
            del names[i]
        else:
            i += 1



def main(catdata, sampling_rate, decoder,
         outfile,
         min_syl_ms=default_min_syl,
         min_silent_ms=default_min_silent):
    from pandas import DataFrame
    min_syl = min_syl_ms / 1000
    min_silent = min_silent_ms / 1000
    start, stop, name = first_pass(np.argmax(catdata, 1), decoder, 1/sampling_rate)
    second_pass(start, stop, name, min_silent)
    third_pass(start, stop, name, min_syl)
    bark.write_events(outfile,
                      DataFrame(dict(start=start,
                                        stop=stop,
                                        name=name)),
                      columns={'start': {'units': 's'},
                          'stop': {'units': 's'},
                          'name': {'units': None}})

def _run():
    ''' Function for getting commandline args.'''
    import argparse

    p = argparse.ArgumentParser(description='''
    Create a segment label file from a 2D categorical probability series

    Uses method from Koumura & Okanoya 2016.

    First the most likely syllable is created.
    Then from these threshold crossings, any short gaps are annealed,
    and any short syllables are removed.
    ''')
    p.add_argument('cat', help='name of a sampled dataset')
    p.add_argument('out',
                   help='name of output event dataset')
    p.add_argument('--min-syl',
                   help='minimum syllable length in ms, default: {}'
                   .format(default_min_syl),
                   type=int,
                   default=default_min_syl)
    p.add_argument('--min-silent',
                   help='minimum silence length in ms, default: {}'
                   .format(default_min_silent),
                   type=int,
                   default=default_min_silent)
    args = p.parse_args()
    sampled = bark.read_sampled(args.cat)
    sr = sampled.sampling_rate
    decoder = sampled.attrs['decoder']
    main(sampled.data, sr, decoder, args.out, args.min_syl,
         args.min_silent)


if __name__ == '__main__':
    _run()
