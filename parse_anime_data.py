if __name__ == '__main__':
    #import multiprocessing as mp
    #mp.freeze_support()
    from dask.distributed import Client
    client = Client()

from ast import literal_eval
import os
import re
import dask.dataframe as dd
from dask_ml.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


if len(os.listdir('./animedata/')) == 0:  # If parquet folder is empty
    print('Processing CSV for use...')
    print('Reading CSV...')
    df = dd.read_csv('original_data/anime.csv',  
                     parse_dates=['aired_from', 'aired_to'],
                     dtype={'mal_id': 'int64',
                            'title': 'object',
                            'type': 'category',
                            'score': 'float64',
                            'scored_by': 'float64',
                            'status': 'category',
                            'episodes': 'float64',
                            'aired_from': 'object',
                            'aired_to': 'object',
                            'source': 'category',
                            'members': 'float64',
                            'favorites': 'float64',
                            'duration': 'object',
                            'rating': 'category',
                            'nsfw': 'category',
                            'pending_approval': 'category',
                            'premiered_season': 'category',
                            'premiered_year': 'float64',
                            'broadcast_day': 'category',
                            'broadcast_time': 'object',
                            'genres': 'object',
                            'themes': 'object',
                            'demographics': 'object',
                            'studios': 'object',
                            'producers': 'object',
                            'licensors': 'object',
                            'synopsis': 'object',
                            'background': 'object',
                            'main_picture': 'object',
                            'url': 'object',
                            'trailer_url': 'object',
                            'title_english': 'object',
                            'title_japanese': 'object',
                            'title_synonyms': 'object',
                            }).set_index('mal_id')


    print('Dropping unused columns...')
    TO_DROP = ['title', 'aired_to', 'nsfw', 'premiered_season',
               'premiered_year', 'broadcast_day', 'broadcast_time', 
               'background', 'main_picture', 'url', 'trailer_url', 
               'title_english', 'title_japanese', 'title_synonyms']
    df = df.drop(columns=TO_DROP)


    print('Filtering entries...')
    # Drop entries with given empty columns (for masking)
    df = df.dropna(subset=['type', 
                           'status',
                           'pending_approval',
                           'duration',
                           ])
    # Mask entries that haven't finished
    df = df.mask(df['status'] != 'Finished Airing')
    # Mask entries not approved yet
    df = df.mask(df['pending_approval'] == 'TRUE')
    # Mask music, not interested in that
    df = df.mask(df['type'] == 'Music')
    # Drop all entries with null values, including the ones masked previously
    df = df.dropna()  #subset=['status', 'pending_approval', 'type'])
    # Drop columns which are now useless
    df = df.drop(columns=['status', 'pending_approval'])


    print('Converting dates...')
    # Turn airing_from dates into days since the first show started airing
    df['aired_from'] = ((df['aired_from'] - df['aired_from'].min()) 
                        / np.timedelta64(1, 'D'))
    # TODO: Add categorical variables for different seasons / weekdays / etc?


    print('Parsing durations...')
    # Parse durations into seconds
    rex = re.compile(r'''((?P<hours>\d*)\ hr\ ?)?
                         ((?P<mins>\d*)\ min\ ?)?
                         ((?P<secs>\d*)\ sec)?
                         (\ per\ ep)?''',
                        re.VERBOSE)
    
    def parse_duration(duration):
        out = re.search(rex, duration).groupdict()
        return (int(out['hours'] or 0)*60*60 
              + int(out['mins'] or 0)*60 
              + int(out['secs'] or 0))
    
    df['duration'] = df['duration'].apply(parse_duration, 
                                          meta=('duration', 'int64'))


    print('Encoding lists...')
    LISTS = ['genres', 'themes', 'demographics', 'studios', 'producers', 
             'licensors']
    mlb = MultiLabelBinarizer()
    df = df.compute()
    for col in LISTS:
        # Evaluate strings into lists and compute
        df[col] = df[col].apply(literal_eval)
        # Transform to new many-hot encoded columns
        mlb.fit(df[col])
        new_col_names = [f"{col}_{x}" for x in mlb.classes_]
        new_cols = pd.DataFrame(data=mlb.fit_transform(df[col]),
                                columns=new_col_names,
                                index=df[col].index,
                                dtype='int64')
        # Join the new columns
        df = df.join(new_cols)
    # Convert back to Dast DataFrame and drop old columns
    df = dd.from_pandas(df, npartitions=1)
    df = df.drop(columns=LISTS)


    print('Encoding categorical features...')
    CATEGORICAL = ['type', 'source', 'rating']
    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(df[CATEGORICAL])
    df = df.join(encoded)
    df = df.drop(columns=CATEGORICAL)


    # TODO: handle synopsis (TfidfVectorizer?)


    #print('Computing DataFrame...')
    #df = df.compute()

    print('Writing to Parquet...')
    df.to_parquet('./animedata/')
    print("Done!")
