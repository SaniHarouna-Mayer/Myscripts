from myscripts.tools import *
from myscripts.helper import join_result
import pandas as pd
import os


def test_to_report():
    os.chdir('/Users/sst/project/analysis/19st_clayrock/')
    csv_df = pd.read_csv('meta2/csv.csv')
    fgr_df = pd.read_csv('meta2/fgr.csv')
    joined_df = csv_df.merge(fgr_df, how='left', on='recipe_id').iloc[2:5]
    res_df = join_result(joined_df['csv_file'], column_names=joined_df['data_id'])
    phases = joined_df.iloc[2]['phases']
    print(to_report(res_df, phases, exclude=['dolomite']))
    return


if __name__ == "__main__":
    test_to_report()