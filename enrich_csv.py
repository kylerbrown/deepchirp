import bark
import numpy as np


def strip_tiers(df, song_tier):
    if 'tier' in df:
        df = df[df.tier == song_tier]
    return df


def shorten_and_lowercase_names(df):
    df = df[df.name != '']
    df.names = df.name.str[0]
    df.names = df.name.str.lower()
    return df


def remove_noise_samples(df, noise_name):
    '''discard noise intervals until the number of noise interval
    equals the mean of the number of other classes'''
    n_noise = len(df.name[(df.name == noise_name)])
    noisefreedf = df[df.name != noise_name]
    avg_class_count = int(noisefreedf.name.groupby(noisefreedf.name).count(
    ).mean())
    n_drop = n_noise - avg_class_count
    if n_drop < 0:
        return df
    noise_locs = np.array(df.index[df.names == noise_name])
    np.random.shuffle(noise_locs)
    drop_noise_ix = noise_locs[:n_drop]
    return df.drop(df.index[drop_noise_ix])


def main(in_csv, out_csv, noise_name='z', song_tier=None):
    dset = bark.read_events(in_csv)
    df = dset.data
    if song_tier:
        df = strip_tiers(df, song_tier)
    df = shorten_and_lowercase_names(df)
    df = remove_noise_samples(df, noise_name)
    bark.write_events(out_csv, df, **dset.attrs)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description="""enrich a csv for training a neural network,
                       simplifies labels to first letter and
                       discards most noise labels""")
    p.add_argument("incsv")
    p.add_argument("outcsv")
    p.add_argument("-n", "--noise", help="noise label, default: z",
            default='z')
    p.add_argument("-t", "--tier", help="song tier (from praat if needed)")
    args = p.parse_args()
    main(args.incsv, args.outcsv, args.noise, args.tier)
