from pathlib import Path
import pandas as pd


def simply_loader(args):
    path_ = Path(args.data_path)
    if path_.is_dir():
        data = pd.DataFrame()

        for path in path_.iterdir():
            if args.gen_name is None:
                tmp_df = pd.read_csv(path)

            elif args.gen_name in str(path):
                tmp_df = pd.read_csv(path)

            data = data.append(tmp_df)

    elif path_.is_file():
        data = pd.DataFrame(path_)

    data.reset_index(inplace=True, drop=True)

    if args.location == 'westsouth':
        data.columns = ['DateTime', 'WindD1', 'WindD2', 'ActivePower', 'WindSpeed', 'RotorPosition',
                        'BladePitchP1', 'BladePitchP2', 'BladePitchP3', 'GenertorTR', 'GenertorTRD']

        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data[['DateTime', 'WindD1', 'WindD2', 'WindSpeed', 'RotorPosition',
                     'BladePitchP1', 'BladePitchP2', 'BladePitchP3', 'GenertorTR', 'ActivePower']]

    elif args.location == 'jeju':
        data['datetime'] = pd.to_datetime(data['datetime'])

    return data
