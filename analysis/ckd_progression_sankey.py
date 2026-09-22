import os

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from paths import RAW_ROOT

df = pd.read_csv(os.path.join(RAW_ROOT, 'DM_df_filtered', 'rj_bf_diag', 'DM_all.csv'), index_col=0)

df = df[~pd.isnull(df['eGFR'])]

# assign CKD stage labels
df['CKD_stage'] = np.where(df['eGFR'] >= 90, 1, 0)
df['CKD_stage'] = np.where((df['eGFR'] < 90) & (df['eGFR'] >= 60), 2, df['CKD_stage'])
df['CKD_stage'] = np.where((df['eGFR'] < 60) & (df['eGFR'] >= 45), 3, df['CKD_stage'])
df['CKD_stage'] = np.where((df['eGFR'] < 45) & (df['eGFR'] >= 30), 4, df['CKD_stage'])
df['CKD_stage'] = np.where((df['eGFR'] < 30) & (df['eGFR'] >= 15), 5, df['CKD_stage'])
df['CKD_stage'] = np.where(df['eGFR'] < 15 , 6, df['CKD_stage'])

df[df['CKD_stage'] == 0]['eGFR']

# ## yearly aggregation

yearly_df = df.groupby(['num', 'year_gap'])[['eGFR', 'CKD_stage']].agg('first').reset_index()

yearly_df = yearly_df[yearly_df['year_gap'] <= 10]
yearly_df.sort_values(by=['num', 'year_gap'], inplace=True)

# reject patients with a single data point
num_data_per_patient = yearly_df.groupby(['num'])['CKD_stage'].count().reset_index()
valid_patient = num_data_per_patient[num_data_per_patient['CKD_stage'] > 1]['num'].to_list()

yearly_df = yearly_df[yearly_df['num'].isin(valid_patient)]

# extract next CKD stages
yearly_df['next_CKD_stage'] = yearly_df.groupby(['num'])['CKD_stage'].shift(-1)

# reject data with the gap larger than 1 year
yearly_df.rename(columns={'year_gap': 'year'}, inplace=True)
yearly_df['year_gap'] = yearly_df.groupby(['num'])['year'].diff(-1)
yearly_df_all = yearly_df[yearly_df['year_gap'] == -1]

# reject patients without the data at 0-th year and data with the gap larger than 1 year
starting_year = yearly_df_all.groupby(['num'])['year'].agg('first').reset_index()
valid_patient = starting_year[starting_year['year'] == 0]['num'].to_list()

yearly_df_pure = yearly_df_all[yearly_df_all['num'].isin(valid_patient)]

yearly_df_pure.head(20)

# ## Deindividualization

def convert2integstage(stage):
    if stage > 3:
        return stage - 2
    elif stage > 1:
        return stage - 1
    else:
        return stage

count_arr_all = np.zeros((10, 6, 6)) # year, stage, stage 
count_arr_pure = np.zeros_like(count_arr_all)
# the element at (i, j, k) indicates that the number of patients who were stage j at i-th year and developed to stage k

integ_count_arr_all = np.zeros((10, 4, 4)) # integrated stages 1, 2 and stages 3a, 3b
integ_count_arr_pure = np.zeros_like(integ_count_arr_all)

yearly_df_all['next_CKD_stage'] = yearly_df_all['next_CKD_stage'].astype('int32')
yearly_df_pure['next_CKD_stage'] = yearly_df_pure['next_CKD_stage'].astype('int32')

# all
for _, d in yearly_df_all.iterrows():
    count_arr_all[d['year'], d['CKD_stage']-1, d['next_CKD_stage']-1] += 1
    integ_count_arr_all[d['year'], convert2integstage(d['CKD_stage'])-1, convert2integstage(d['next_CKD_stage'])-1] += 1
   
# pure
for _, d in yearly_df_pure.iterrows():
    count_arr_pure[d['year'], d['CKD_stage']-1, d['next_CKD_stage']-1] += 1
    integ_count_arr_pure[d['year'], convert2integstage(d['CKD_stage'])-1, convert2integstage(d['next_CKD_stage'])-1] += 1

count_arr_all = count_arr_all.astype(int)
count_arr_pure = count_arr_pure.astype(int)
integ_count_arr_all = integ_count_arr_all.astype(int)
integ_count_arr_pure = integ_count_arr_pure.astype(int)

def arr2values(arr):
    num_year, num_stage, _ = arr.shape
    
    source = []
    target = []
    value = []
    
    for y in range(num_year):
        for source_s in range(num_stage):
            for target_s in range(num_stage):
                source.append(y*num_stage + source_s)
                target.append((y+1)*num_stage + target_s)

                if arr[y, source_s, target_s] > 0:
                    value.append(arr[y, source_s, target_s])
                else:
                    value.append(1)
    return source, target, value

def plot_sankey(arr, stage_label, colors=['green', 'lawngreen', 'sandybrown', 'orange', 'tomato', 'red']):
    
    num_year, num_stage, _ = arr.shape
    num_year += 1
    
    assert len(colors) == num_stage
    
    if num_stage == len(stage_label):
        stage_label = stage_label * num_year
    elif num_stage*num_year != len(stage_label):
        raise ValueError('Given stage label is not compatible with the flow array.')
    
    # position
    x = []
    for i in range(num_year):
        x += [i/(num_year-1)]*num_stage
    y = []
    for i in range(num_stage):
        y += [i/(num_stage-1)]
    y = y*(num_year)
    
    # color
    color = []
    alpha = .5
    for i in range(num_year-1):
        for ii in range(num_stage):
            for iii, c in enumerate(colors):
                if arr[i, ii, iii] > 0:
                    color += [[c, alpha]]
                else:
                    color += [['white', 0]]
    color = ['rgba'+str(matplotlib.colors.to_rgba(i[0], i[1])) for i in color]
    
    node_config = dict(
        pad = 10,
        thickness = 5,
        line = dict(color='black', width=0.5),
        label = stage_label,
        color = colors*num_year,
        x = [.001 if v==0 else .999 if v==1 else v for v in x],
        y = [.001 if v==0 else .999 if v==1 else v for v in y]
    )

    s, t, v = arr2values(arr)
    data = dict(
        source = s,
        target = t,
        value = v,
        color = color
    )

    fig = go.Figure(data=[go.Sankey(
        arrangement = 'snap',
        node = node_config,
        link = data, 
    )])

    fig.show()

# Number of patients
prev_count = count_arr_pure[0].sum().sum()
print('Year\tNum. patients\tDiff. Num. Patient')
for yi, arr in enumerate(count_arr_pure):
    current_count = arr.sum().sum()
    print(f'{yi} \t{current_count} \t\t{current_count - prev_count}')
    prev_count = current_count

plot_sankey(count_arr_pure, ['stage 1', 'stage 2', 'stage 3a', 'stage 3b', 'stage 4', 'stage 5'])

plot_sankey(integ_count_arr_pure, ['normal', 'stage 3', 'stage 4', 'stage 5'], colors=['yellowgreen', 'sandybrown', 'orange', 'red'])

