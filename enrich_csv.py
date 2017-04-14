import numpy as np
import pandas as pd
import bark


def strip_tiers(df, song_tier):
    if 'tier' in df:
        df = df[df.tier == song_tier]
    return df


def shorten_and_lowercase_names(df):
    df = df[df.name != '']
    df.names = df.name.str[0]
    df.names = df.name.str.lower()
    return df


def add_boundaries(df, boundary_size=0.03, boundary_label='__'):
    labels = df.to_dict('records')
    i = 0
    while i < len(labels):
        # left boundary
        if labels[i]['name'] == boundary_label:
            pass
        elif i == 0:
            boundary = max(0, labels[i]['start'] - boundary_size)
            labels.insert(0,
                          dict(start=boundary,
                               stop=labels[i]['start'],
                               name=boundary_label))
        elif labels[i]['start'] - labels[i - 1]['stop'] == 0:
            pass
        elif labels[i]['start'] - labels[i - 1]['stop'] < boundary_size:
            labels.insert(i,
                          dict(start=labels[i - 1]['stop'],
                               stop=labels[i]['start'],
                               name=boundary_label))
        else:
            labels.insert(i,
                          dict(start=labels[i]['start'] - boundary_size,
                               stop=labels[i]['start'],
                               name=boundary_label))
        # right boundary
        if labels[i]['name'] == boundary_label:
            pass
        elif i == len(labels) - 1:
            labels.insert(i + 1,
                          dict(start=labels[i]['stop'],
                               stop=labels[i]['stop'] + boundary_size,
                               name=boundary_label))
        elif labels[i + 1]['start'] - labels[i]['stop'] < boundary_size:
            pass  # covered by left boundary on next loop
        else:
            labels.insert(i + 1,
                          dict(start=labels[i]['stop'],
                               stop=labels[i]['stop'] + boundary_size,
                               name=boundary_label))
        i += 1
    return pd.DataFrame(labels)


def remove_noise_samples(df, noise_name):
    '''discard noise intervals until the number of noise interval
    equals the mean of the number of other classes'''
    n_noise = len(df.name[(df.name == noise_name)])
    noisefreedf = df[df.name != noise_name]
    print(noisefreedf.name.groupby(noisefreedf.name).count())
    avg_class_count = int(noisefreedf.name.groupby(noisefreedf.name).count(
    ).mean())
    n_drop = n_noise - avg_class_count
    if n_drop < 0:
        return df
    noise_locs = np.array(df.index[df.names == noise_name])
    np.random.shuffle(noise_locs)
    drop_noise_ix = noise_locs[:n_drop]
    return df.drop(df.index[drop_noise_ix])


def main(in_csv,
         out_csv,
         noise_name='z',
         song_tier=None,
         boundary_length=0.00,
         boundary_label='__'):
    dset = bark.read_events(in_csv)
    df = dset.data
    if song_tier:
        df = strip_tiers(df, song_tier)
    df = shorten_and_lowercase_names(df)
    df = remove_noise_samples(df, noise_name)
    if boundary_length > 0:
        df = add_boundaries(df, boundary_size=boundary_length,
                boundary_label=boundary_label)
    bark.write_events(out_csv, df, **dset.attrs)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description="""enrich a csv for training a neural network,
                       simplifies labels to first letter and
                       discards most noise labels""")
    p.add_argument("incsv")
    p.add_argument("outcsv")
    p.add_argument("-n",
                   "--noise",
                   help="noise label, default: z",
                   default='z')
    p.add_argument("-t", "--tier", help="song tier (from praat if needed)")
    p.add_argument("-b",
                   "--boundary",
                   help="set boundary length, units: seconds",
                   type=float,
                   default=0.00)
    p.add_argument("-l",
                   "--boundary-label",
                   help="boundary label, default: __",
                   default='__')


    args = p.parse_args()
    main(args.incsv, args.outcsv, args.noise, args.tier, args.boundary,
            args.boundary_label)
