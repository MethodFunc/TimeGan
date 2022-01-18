def location_preprocessing(dataframe, args):
    df = dataframe.copy()
    if args.location == 'westsouth':
        df.set_index('DateTime', inplace=True)
        df['ActivePower'] = df.apply(
            lambda x: 0 if (x['WindSpeed'] > 3) and (x['ActivePower'] <= 0) else x['ActivePower'], axis=1)
        df['ActivePower'] = df['ActivePower'].apply(lambda x: 0 if x < 0 else x)

    elif args.location == 'jeju':
        df.set_indeX('datetime', inplace=True)
        df['INFO_POWER_ACTIVE'] = df['INFO_POWER_ACTIVE'].map(lambda x: 0 if x < 0 else x)

    df = df.resample(args.freq).mean()

    df.dropna(inplace=True)

    return df
