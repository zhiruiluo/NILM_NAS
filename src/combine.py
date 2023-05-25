from __future__ import annotations

import func
import pandas as pd


def main():
    new = False
    ts_per_instance = 60
    threshold = 10
    main_meter = 'mains1'
    labels = pd.read_csv(
        'low_freq/house_2/labels.dat', sep=' ', header=None, index_col=0, squeeze=True,
    )

    if new is True:
        main1 = pd.read_csv(
            'low_freq/house_2/channel_1.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        main2 = pd.read_csv(
            'low_freq/house_2/channel_2.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df = pd.DataFrame(columns=labels, index=main1.index)
        df['mains1'] = main1
        df['mains2'] = main2
        ko1 = pd.read_csv(
            'low_freq/house_2/channel_3.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['kitchen_outlets1'] = ko1
        lighting = pd.read_csv(
            'low_freq/house_2/channel_4.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['lighting'] = lighting
        stove = pd.read_csv(
            'low_freq/house_2/channel_5.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['stove'] = stove
        microwave = pd.read_csv(
            'low_freq/house_2/channel_6.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['microwave'] = microwave
        washer_dryer = pd.read_csv(
            'low_freq/house_2/channel_7.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['washer_dryer'] = washer_dryer
        ko2 = pd.read_csv(
            'low_freq/house_2/channel_8.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['kitchen_outlets2'] = ko2
        refrigerator = pd.read_csv(
            'low_freq/house_2/channel_9.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['refrigerator'] = refrigerator
        dishwaser = pd.read_csv(
            'low_freq/house_2/channel_10.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['dishwaser'] = dishwaser
        disposal = pd.read_csv(
            'low_freq/house_2/channel_11.dat',
            sep=' ',
            header=None,
            index_col=0,
            squeeze=True,
        )
        df['disposal'] = disposal
        start_row, end_row = ko1.index[0], ko1.index[-1]

        df = df[start_row <= df.index]
        df = df[df.index <= end_row + 3]
        df = df.reset_index(drop=True)
        df.to_csv('combine_h2.csv')
    else:
        df = pd.read_csv('combine_h2.csv', index_col=0)
    # slice a small amount of the dataset for testing purpose:
    # df = df.head(12000)
    out_columns = list(range(ts_per_instance)) + labels[2:].tolist()
    out_df = pd.DataFrame(columns=out_columns)
    group_df = df.groupby(df.index // ts_per_instance)
    for i, g in group_df:
        for ii, d in enumerate(g[main_meter].items()):
            out_df.at[i, ii] = d[1]
        g = g.iloc[:, 2:]
        for app_name, appliance in g.iteritems():
            out_df.at[i, app_name] = func.power_config(
                data=appliance, threshold=threshold,
            )
    out_df.to_csv('out_df.csv')


if __name__ == '__main__':
    main()
