import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape_score

from typing import Tuple

MONTH2NUM = {m: (e+1)/12 for e, m in enumerate([
    'JAN','FEB','MAR','APR','MAY','JUN',
    'JUL','AUG','SEP','OCT','NOV','DEC'
])}


def load_all_csvs(state = 'IOWA') -> pd.DataFrame:
    """
    hardcoded loader of csvs
    """
    df_harvest, df_production, df_yield = [
        add_features(load_dataframe(fn, state=state), value_type=vt)
        for fn, vt in zip([
            'usda_harvested_acreage_subset.csv',
            'usda_production_data.csv',
            'usda_yield_data_subset.csv'
        ], ['harvest', 'prod', 'yield'])
    ]
    
    return df_harvest, df_production, df_yield


def join_csvs_and_filter_by_year(
    df_tup: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    """
    Join dataframes with index Year. filter by YEAR
    """
    df_harvest, df_production, df_yield = df_tup
    
    df_harvest = df_harvest[df_harvest['Period'] == 'YEAR']
    df_production = df_production[df_production['Period'] == 'YEAR']
    df_yield = df_yield[df_yield['Period'] == 'YEAR']
    
    df_harvest = df_harvest.set_index('Year')
    df_production = df_production.set_index('Year')
    df_yield = df_yield.set_index('Year')

    df_target = pd.concat([
        df_harvest.groupby(level=0).sum(),
        df_production.groupby(level=0).sum(),
        df_yield.groupby(level=0).mean()
    ], axis=1)
    
    return df_target


def load_dataframe(
    filename: str,
    state: str = 'IOWA'
) -> pd.DataFrame:
    """
    """
    df = pd.read_csv(filename)
    
    df = df.loc[
        (df['Commodity'] == 'CORN') & \
        (df['State'] == state),
        [
            'Year', 
            'Period',
            'State',
            'Commodity',
            'Data Item',
            'Value',
            'CV (%)'
        ]
    ]
    return df

def norm_a_column(
    pds: pd.core.series.Series,
) -> pd.core.series.Series:
    
    eps = 0.0001
    
    pds = \
        (pds-pds.min())/ \
        (pds.max()-pds.min()+eps)
    
    return pds


def unnorm_a_column(
    pds,
    original_pds,
):
    
    eps = 0.0001
    
    range_plus_eps = (original_pds.max()-original_pds.min()+eps)
    
    return pds*range_plus_eps + original_pds.min()


def norm_code_from_month(period: str) -> float:
    """
    """
    lst = [
        MONTH2NUM[m]
        for m in MONTH2NUM
        if m in period
    ]

    if len(lst) == 0:
        ncode = 0.0
    else:
        ncode = lst[0]

    return ncode


def add_features(
    df: pd.DataFrame,
    value_type: str,
) -> pd.DataFrame:
    """
    N as a prefix stands for normalized
    """
    df['NYear'] = (df['Year']-2000)/25  # Year between 2000 and 2025
    df['NPeriod'] = df['Period'].apply(norm_code_from_month)
    df['NYP'] = df['NYear'] + df['NPeriod']/12
    df[f'{value_type}'] = df['Value']
    df[f'N{value_type}'] = norm_a_column(df['Value'])
    
    
    return df


def load_all_jsons(state: str = 'Iowa'):
    """
    """
    with open('lst_subset_august.json', 'r') as src:
        data = json.load(src)

    lst_lst = [
        (d['year'], float(d['LST day corn (avg)']))
        for d in data['data']
        if \
        d['state'] == state and \
        d['LST day corn (avg)'] != 'nan'
    ]

    with open('ndvi_subset_august.json', 'r') as src:
        data = json.load(src)

    ndvi_lst = [
        (d['year'], float(d['NDVI corn (avg)']))
        for d in data['data']
        if \
        d['state'] == state and \
        d['NDVI corn (avg)'] != 'nan'
    ]

    with open('nldas_subset_august.json', 'r') as src:
        data = json.load(src)

    meteo_lst = [
        (
            d['year'],
            float(d['precipitation corn (avg)']),
            float(d['temperature corn (avg)']),
            float(d['vpd corn (avg)'])
        )
        for d in data['data']
        if (
            d['state'] == state and \
            d['precipitation corn (avg)'] != 'nan' and \
            d['temperature corn (avg)'] != 'nan' and \
            d['vpd corn (avg)'] != 'nan'
        )
    ]
    
    df = pd.DataFrame(
        [el[1] for el in lst_lst],
        index =[el[0] for el in lst_lst],
        columns =['LST']
    )
    
    # To dataframe
    
    df = pd.concat([
        pd.DataFrame(
            [el[1] for el in lst_lst],
            index =[el[0] for el in lst_lst],
            columns =['LST']
        ).groupby(level=0).mean(),
        pd.DataFrame(
            [el[1] for el in ndvi_lst],
            index =[el[0] for el in ndvi_lst],
            columns =['ndvi']
        ).groupby(level=0).mean(),
        pd.DataFrame(
            [el[1:] for el in meteo_lst],
            index =[el[0] for el in meteo_lst],
            columns =['pr', 'temp', 'vpd']
        ).groupby(level=0).mean()
    ], axis=1)
    
    # Normalized features:
    
    for col in ['LST', 'ndvi', 'pr', 'temp', 'vpd']:
        df[f'N{col}'] = norm_a_column(df[f'{col}'])
    
    df['NYear'] = (df.index-2000)/25  # Year between 2000 and 2025
    
    return df


def get_scores(y_test, y_pred, df_target):
    """
    order is always: harvest(X), prod, yied
    
    Args:
    
        y_test (array): target normalized array
        y_pred (array): predicted normalized array
        df_target (pd.DataFrame): dataframe of reference for undoing normalization
        
    Returns:
    
        Dict[str, float]: prod,yield list of metrics r2, MAPE 
    """
    # Undo normalization:
    
    predicted_arr = np.asarray([
        unnorm_a_column(
            pds = y_pred[:,0],
            original_pds = df_target['prod'].to_numpy()
        ),
        unnorm_a_column(
            pds = y_pred[:,1],
            original_pds = df_target['yield'].to_numpy()
        )
    ]).T

    target_arr = np.asarray([
        unnorm_a_column(
            pds = y_test[:,0],
            original_pds = df_target['prod'].to_numpy()
        ),
        unnorm_a_column(
            pds = y_test[:,1],
            original_pds = df_target['yield'].to_numpy()
        )
    ]).T
    
    # Metrics
    
    r2, mape = [], []
    for e, _ in enumerate([
        #'harvest',
        'prod',
        'yield'
    ]):
        r2.append(float(
            r2_score(y_test[e], y_pred[e], multioutput='raw_values')
        ))
        
        # log to mlflow?
        mape.append(float(
            100*mape_score(target_arr[e], predicted_arr[e], multioutput='raw_values')
        ))
        
    return {'r2': r2, 'mape': mape}

