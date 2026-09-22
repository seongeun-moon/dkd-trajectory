import os
import shutil

import numpy as np
import pandas as pd

from paths import RAW_ROOT

    

    

    

def median_or_last_value(dataframe):
    def custom_function(x):
        if len(x) > 2:
            return x.median()
        else:
            return x.iloc[-1]

    re_est = [k for k in list(dataframe.keys()) if 're_est' in k]
    dataframe['final'] = dataframe.groupby(['num', 'date'])[re_est].transform(custom_function)
    return dataframe
    
    
def agg_duplicates(df, column_name):
    # one original estimation and one re-estimation => re-estimated value
    # one original estimation and multiple re-estimation => median of re-estimated values
    # multiple original estimation (and one re-estimation) => median
    # two original estimation => mean

    df.drop_duplicates(inplace=True)
    
    df[column_name] = df[column_name].astype(str)
    df['re_est'] = df[column_name].str.contains('재검', na=False)
    df['re_est_count'] = df.groupby(['num', 'date'])['re_est'].transform(sum)

    df_count = df.groupby(['num', 'date'])['re_est'].count().reset_index().rename(columns={'re_est': 'count'})
    df = pd.merge(df, df_count, on=['num', 'date'])
    
    df.loc[df['re_est'], column_name+'_re_est'] = df[df['re_est']][column_name].str.extract(r'(?:[><]?)([\.\d]*)').apply(pd.to_numeric, errors='coerce').values
    df[column_name] =  df[column_name].str.extract(r'(?:[><]?)([\.\d]*)').apply(pd.to_numeric, errors='coerce').values
    df['final'] = df[column_name].copy()
    df = df[~(pd.isnull(df[column_name]) & pd.isnull(df[column_name+'_re_est']))]

    # zero original estimation and more than two re-estimation => median of re-estimations
    # zero original estimation and two re-estimation => last re-estimation
    # zero original estimation and one re-estimation => re-estimation
    cdt = (df['re_est_count'] > 0) & (df['re_est_count'] == df['count'])
    if sum(cdt) > 0:
        df_value = df[cdt][['num', 'date', column_name, column_name+'_re_est']]
        df_value[column_name+'_re_est'].fillna(df[column_name], inplace=True)
        df_value[column_name+'_re_est'] = df_value[column_name+'_re_est'].astype(float)
        df_value = median_or_last_value(df_value)
        df_value = df_value[['num', 'date', 'final']].drop_duplicates()
        
        df = pd.merge(df, df_value, on=['num', 'date'], how='left')
        if 'final_x' in df.keys():
            df['final'] = np.where(pd.isnull(df['final_y']), df['final_x'], df['final_y'])
            df.drop(['final_x', 'final_y'], axis=1, inplace=True)
        df.drop_duplicates(['num', 'date', 'final'], inplace=True)
    
    
    # one original estimation and one re-estimation => re-estimation
    cdt = (df['re_est_count'] == 1) & (df['count'] == 2)
    if sum(cdt) > 0:
        df_value = df[cdt & df['re_est']][['num', 'date', column_name+'_re_est']].rename(columns={column_name+'_re_est': 'final'})
        df = pd.merge(df, df_value, on=['num', 'date'], how='left')
        if 'final_x' in df.keys():
            df['final'] = np.where(pd.isnull(df['final_y']), df['final_x'], df['final_y'])
            df.drop(['final_x', 'final_y'], axis=1, inplace=True)
        df.drop_duplicates(['num', 'date', 'final'], inplace=True)
    
    # one original estimation and multiple re-estimation => median of re-estimations
    cdt = (df['re_est_count'] > 1) & (df['count'] == df['re_est_count'] + 1)
    if sum(cdt) > 0:
        df_value = df[cdt & df['re_est']][['num', 'date', column_name+'_re_est']]
        df_value['final'] = df_value.groupby(['num', 'date'])[column_name+'_re_est'].transform('median')
        df_value = df_value[['num', 'date', 'final']].drop_duplicates()

        df = pd.merge(df, df_value, on=['num', 'date'], how='left')
        if 'final_x' in df.keys():
            df['final'] = np.where(pd.isnull(df['final_y']), df['final_x'], df['final_y'])
            df.drop(['final_x', 'final_y'], axis=1, inplace=True)
        df.drop_duplicates(['num', 'date', 'final'], inplace=True)

    # multiple original estimation (and zero, one or multiple re-estimation) => median
    # two original estimation => last
    cdt = ((df['count'] - df['re_est_count'] >= 2)) | ((df['re_est_count'] == 0) & (df['count'] == 2))
    if sum(cdt) > 0:
        df_value = df[cdt][['num', 'date', column_name, column_name+'_re_est']]
        df_value[column_name+'_re_est'].fillna(df[column_name], inplace=True)
        df_value[column_name+'_re_est'] = df_value[column_name+'_re_est'].astype(float)
        df_value = median_or_last_value(df_value)
        df_value = df_value[['num', 'date', 'final']].drop_duplicates()
        
        df = pd.merge(df, df_value, on=['num', 'date'], how='left')
        if 'final_x' in df.keys():
            df['final'] = np.where(pd.isnull(df['final_y']), df['final_x'], df['final_y'])
            df.drop(['final_x', 'final_y'], axis=1, inplace=True)
        df.drop_duplicates(['num', 'date', 'final'], inplace=True)
    
    try:
        assert df.drop_duplicates(['num', 'date']).shape[0] == df.drop_duplicates(['num', 'date', 'final']).shape[0]
    except AssertionError:
        raise
    
    df['final'].fillna(df[column_name], inplace=True)
    df.drop([column_name, 're_est', 're_est_count', 'count', column_name + '_re_est'], axis=1, inplace=True)
    df.rename(columns={'final': column_name}, inplace=True)
    
    return df.drop_duplicates()
        

    

def load_vital(name, df_path, vital_list):
    """Concatenate every DM_vital_<name>* export into one frame."""
    files = [f for f in vital_list if f.startswith(f'DM_vital_{name}')]
    df = pd.read_csv(os.path.join(df_path, files[0]))
    for f in files[1:]:
        df = pd.concat((df, pd.read_csv(os.path.join(df_path, f))), axis=0)
    df['date'] = df['date'].str.split(' ', expand=True)[0]
    df[name] = df[name].astype(str)
    return df

def save_vital(df, name, df_filtered_path):
    """Median-aggregate per patient-day and write the three-column export."""
    df[name] = df.groupby(['num', 'date'])[name].transform(np.median)
    df.drop_duplicates(inplace=True)
    df[['num', 'date', name]].to_csv(os.path.join(df_filtered_path, f'DM_vital_{name}.csv'))


def main():
    data_path = RAW_ROOT
    hospital = 'snubh'
    session = '2'
    base_path = os.path.join(data_path, hospital, session)
    df_path = os.path.join(base_path, 'DM_df')
    df_filtered_path = os.path.join(base_path, 'DM_df_filtered')
    reject_data_before_diagnosis = True
    os.makedirs(df_filtered_path, exist_ok=True)
    data_list = os.listdir(df_path)
    ## CAUTION: it removes all files under the df_filtered_path ##
    for f in os.listdir(df_filtered_path):
        if os.path.isfile(os.path.join(df_filtered_path, f)):
            os.remove(os.path.join(df_filtered_path, f))
    for f in data_list:
        if os.path.isfile(os.path.join(df_path, f)):
            shutil.copy(os.path.join(df_path, f), os.path.join(df_filtered_path, f))
    # ### Filtering abnormal data
    # #### Creatinine_serum
    # #### Glucose
    # #### Potassium
    # #### Sodium
    # #### RBC_urine
    # #### UA_Albumin
    try:
        filename = 'DM_lab_10_Creatinine_serum.csv'
        df = pd.read_csv(os.path.join(df_path, filename))
    except FileNotFoundError:
        filename = 'DM_lab_Creatinine_serum.csv'
        df = pd.read_csv(os.path.join(df_path, filename), dtype='object')
    null_list = ['(body F.)', '(bod F.)', '(j-p)', '(drain)', '>', '<']
    for i in null_list:
        try:
            cdt = df['Creatinine_serum'].str.contains(i, na=False)
        except AttributeError:
            raise
        df['Creatinine_serum'][cdt] = np.nan
    df.to_csv(os.path.join(df_filtered_path, filename), index=False)
    try:
        filename = 'DM_lab_16_Glucose.csv'
        df = pd.read_csv(os.path.join(df_path, filename))
    except FileNotFoundError:
        filename = 'DM_lab_Glucose.csv'
        df = pd.read_csv(os.path.join(df_path, filename), dtype='object')
    null_list = ['(body F.)']
    for i in null_list:
        cdt = df['Glucose'].str.contains(i, na=False)
        df['Glucose'][cdt] = np.nan
    df.to_csv(os.path.join(df_filtered_path, filename), index=False)
    try:
        filename = 'DM_lab_29_Potassium.csv'
        df = pd.read_csv(os.path.join(df_path, filename))
    except FileNotFoundError:
        filename = 'DM_lab_Potassium.csv'
        df = pd.read_csv(os.path.join(df_path, filename), dtype='object')
    null_list = ['hemo', 'Hemo', 'HEMO']
    reserve_list = ['nonhemolysis']
    for i in null_list:
        cdt = df['Potassium'].str.contains(i, na=False)
        reserve_cdt = df['Potassium'].str.contains(reserve_list[0], na=False)

        cdt = cdt & (~reserve_cdt)
        df['Potassium'][cdt] = np.nan
    df['Potassium'][df['Potassium'].str.contains('hemo', na=False)]
    df.to_csv(os.path.join(df_filtered_path, filename), index=False)
    try:
        filename = 'DM_lab_33_RBC_urine.csv'
        df = pd.read_csv(os.path.join(df_path, filename))
    except FileNotFoundError:
        filename = 'DM_lab_RBC_urine.csv'
        df = pd.read_csv(os.path.join(df_path, filename), dtype='object')
    print(df['RBC_urine'].value_counts())
    rbc_col = df.columns.get_loc('RBC_urine')
    for i in range(df.shape[0]):
        try:
            v = float(df['RBC_urine'].iloc[i])
            if v < 1:
                df.iloc[i, rbc_col] = 0
            elif v <= 4:
                df.iloc[i, rbc_col] = 1
            elif v <= 9:
                df.iloc[i, rbc_col] = 2
            elif v <= 19:
                df.iloc[i, rbc_col] = 3
            elif v <= 29:
                df.iloc[i, rbc_col] = 4
            elif v <= 49:
                df.iloc[i, rbc_col] = 5
            elif v <= 99:
                df.iloc[i, rbc_col] = 6
            else:
                df.iloc[i, rbc_col] = 7
        except ValueError:
            continue
    RBC_urine_dict = {'< 1': '0',
                      '1 - 4': '1',
                      '1월 4일': '1',
                      '5 - 9': '2',
                      '5월 9일': '2',
                      '10 - 19': '3',
                      '10월 19일': '3',
                      '20 - 29': '4',
                      '30 - 49': '5',
                      '50 - 99': '6',
                      '> 100': '7',
                      '>100': '7',
                      '≥ 100': '7',
                      '≥100': '7',
                      '100': '7'}
    df['RBC_urine'] = df['RBC_urine'].astype(str)
    for key in RBC_urine_dict.keys():
        cdt = df['RBC_urine'].str.contains(key, na=False)
        reserve_cdt = df['RBC_urine'].str.contains('재검', na=False)

        df['RBC_urine'].replace(to_replace=key, value=RBC_urine_dict[key], inplace=True)
        df.loc[cdt & (~reserve_cdt), 'RBC_urine'] = RBC_urine_dict[key]

        if sum(cdt & reserve_cdt) > 0:
            df.loc[cdt & reserve_cdt, 'RBC_urine'] = str(RBC_urine_dict[key]) + df['RBC_urine'][cdt & reserve_cdt].str.extract(r'(\(.*\))').iloc[0].values
    print(df['RBC_urine'].unique())
    df = df[~pd.isnull(df['RBC_urine'])]
    df.to_csv(os.path.join(df_filtered_path, filename), index=False)
    try:
        filename = 'DM_lab_34_Sodium.csv'
        df = pd.read_csv(os.path.join(df_path, filename), dtype='object')
    except FileNotFoundError:
        filename = 'DM_lab_Sodium.csv'
        df = pd.read_csv(os.path.join(df_path, filename), dtype='object')
    df['Sodium'].replace(to_replace='.', value=np.nan, inplace=True)
    reserve_cdt = df['Sodium'].str.contains('재검', na=False)
    df.loc[~reserve_cdt, 'Sodium'] = np.squeeze(df[~reserve_cdt]['Sodium'].str.extract(r'([\.0-9]*)').apply(pd.to_numeric).values)
    df.loc[reserve_cdt, 'Sodium'] = np.squeeze(df['Sodium'][reserve_cdt].str.extract(r'([\.0-9]*)').values) + '(재검)'
    df = df[~pd.isnull(df['Sodium'])]
    df.to_csv(os.path.join(df_filtered_path, filename), index=False)
    try:
        filename = 'DM_lab_39_UA_Albumin.csv'
        df = pd.read_csv(os.path.join(df_path, filename))
    except FileNotFoundError:
        filename = 'DM_lab_UA_Albumin.csv'
        df = pd.read_csv(os.path.join(df_path, filename))
    UA_Albumin_dict = {'-': 0,
                       '-/': 0,
                       '+-/': 1,
                       '+/-': 1,
                       '+/-(재검)': '1(재검)',
                       '+': 2,
                       '1+': 2,
                       '2+': 3,
                       '3+': 4, 
                        '4+': 4}
    for key in UA_Albumin_dict.keys():
        df['UA_Albumin'].replace(to_replace=key, value=UA_Albumin_dict[key], inplace=True)
    df = df[~pd.isnull(df['UA_Albumin'])]
    df.to_csv(os.path.join(df_filtered_path, filename), index=False)
    # ### Microalbumin_cr_ratio
    # no need
    filename = 'DM_lab_microalbumin_cr_ratio.csv'
    # ## Manage duplicated estimations
    df = pd.read_csv(os.path.join(df_path, filename))
    df['microalbumin_cr_ratio'].replace(to_replace='.', value=np.nan, inplace=True)
    reserve_cdt = df['microalbumin_cr_ratio'].str.contains('재검', na=False)
    df.loc[~reserve_cdt, 'microalbumin_cr_ratio'] = np.squeeze(df[~reserve_cdt]['microalbumin_cr_ratio'].str.extract(r'([\.0-9]*)').apply(pd.to_numeric).values)
    if sum(reserve_cdt) > 0:
        df.loc[reserve_cdt, 'microalbumin_cr_ratio'] = np.squeeze(df['microalbumin_cr_ratio'][reserve_cdt].str.extract(r'([\.0-9]*)').values) + '(재검)'
    df = df[~pd.isnull(df['microalbumin_cr_ratio'])]
    df.to_csv(os.path.join(df_path, filename), index=False)
    filtered_data_list = os.listdir(df_filtered_path)
    lab_list = []
    vital_list = []
    for f in filtered_data_list:
        if f.startswith('DM_lab'):
            lab_list.append(f)
        elif f.startswith('DM_vital'):
            vital_list.append(f)
    # #### Lab
    for f in lab_list:
        df = pd.read_csv(os.path.join(df_filtered_path, f))
        lab_name = df.keys()[-1]
        if lab_name in ['microalbumin_cr_ratio', 'cal_Protein_cr_ratio', 'eGFR_CKD-EPI_cal', 'eGFR_cal']:
            continue
        print(lab_name)
        df = agg_duplicates(df, lab_name)
        df.to_csv(os.path.join(df_filtered_path, f), index=False)
    # c-peptide
    if os.path.isfile(os.path.join(df_filtered_path, 'DM_lab_C_peptide_Basal.csv')):

        if os.path.isfile(os.path.join(df_filtered_path, 'DM_lab_C_peptide.csv')):
            df1 = pd.read_csv(os.path.join(df_filtered_path, 'DM_lab_C_peptide_Basal.csv'))
            df2 = pd.read_csv(os.path.join(df_filtered_path, 'DM_lab_C_peptide.csv'))

            df = pd.concat((df1.rename(columns={'C_peptide_Basal': 'C_peptide'}), df2), axis=0).drop_duplicates()
            df['C_peptide'] = df['C_peptide'].astype(float)
            df = df.groupby(['num', 'date'])['C_peptide'].mean().reset_index()

            df.to_csv(os.path.join(df_filtered_path, 'DM_lab_C_peptide.csv'), index=False)
        else:
            os.rename(os.path.join(df_filtered_path, 'DM_lab_C_peptide_Basal.csv'), os.path.join(df_filtered_path, 'DM_lab_C_peptide.csv'))
    # #### Demographics
    demographic_list = ['DM_height.csv', 'DM_weight.csv']
    # #### Vital
    try:
        df_height = pd.read_csv(os.path.join(df_path, 'DM_height.csv'))
        df_weight = pd.read_csv(os.path.join(df_path, 'DM_weight.csv'))

    except FileNotFoundError:
        df_height = pd.read_csv(os.path.join(df_path, 'DM_height_1.csv'))
        df_weight = pd.read_csv(os.path.join(df_path, 'DM_weight_1.csv'))
    df_height['키'] = df_height['키'].astype(str)
    df_height['height'] = df_height['키'].str.extract(r'([\.0-9]*)').apply(pd.to_numeric, errors='coerce')
    df_height['date'] = df_height['date'].str.split(' ', expand=True)[0]
    df_height = df_height[df_height['height'] >= 100][['num', 'date', 'height']]
    df_height.dropna(inplace=True)
    df_height['height'] = df_height.groupby(['num', 'date'])['height'].transform(np.median)
    df_height.drop_duplicates(inplace=True)
    df_height.to_csv(os.path.join(df_filtered_path, 'DM_height.csv'))
    if '몸무게' in df_weight.keys():
        column_name = '몸무게'
    elif '체중' in df_weight.keys():
        column_name = '체중'
    else:
        raise ValueError
    df_weight[column_name] = df_weight[column_name].astype(str)
    df_weight['weight'] = df_weight[column_name].str.extract(r'([\.0-9]*)').apply(pd.to_numeric, errors='coerce')
    df_weight['date'] = df_weight['date'].str.split(' ', expand=True)[0]
    df_weight = df_weight[['num', 'date', 'weight']]
    df_weight.dropna(inplace=True)
    df_weight['weight'] = df_weight.groupby(['num', 'date'])['weight'].transform(np.median)
    df_weight.drop_duplicates(inplace=True)
    df_weight.to_csv(os.path.join(df_filtered_path, 'DM_weight.csv'))
    VITALS = {
        'BT':  (False, (28, None)),
        'DBP': (True,  (30, 250)),
        'SBP': (True,  (30, 250)),
    }
    for name, (drop_zero, (low, high)) in VITALS.items():
        df_vital = load_vital(name, df_path, vital_list)
        if drop_zero:
            df_vital = df_vital[df_vital[name] != '0']
        df_vital[name] = df_vital[name].str.extract(r'([\.0-9]*)').apply(pd.to_numeric, errors='coerce')
        df_vital.dropna(inplace=True)
        df_vital = df_vital[df_vital[name] >= low]
        if high is not None:
            df_vital = df_vital[df_vital[name] <= high]
        save_vital(df_vital, name, df_filtered_path)
    df_pr = load_vital('PR', df_path, vital_list)
    for separator in ['~', '-']:
        cdt = df_pr['PR'].str.contains(separator, na=False)
        if sum(cdt) > 0:
            df_pr.loc[cdt, 'PR'] = df_pr[cdt]['PR'].str.split(separator, expand=True)[1]
    df_pr['PR'] = df_pr['PR'].str.extract(r'([\.0-9]*)').apply(pd.to_numeric, errors='coerce')
    df_pr = df_pr[(df_pr['PR'] > 0) & (df_pr['PR'] <= 300)]
    df_pr.dropna(inplace=True)
    save_vital(df_pr, 'PR', df_filtered_path)
    # ### Prescription
    df = pd.read_csv(os.path.join(df_path, 'DM_prescription.csv'), index_col=0)
    df_agg = df.groupby(['num', 'date'])['code'].apply(list).reset_index()
    df_agg['ingredient'] = df.groupby(['num', 'date'])['ingredient'].apply(list).values
    df_agg['dosage_day_in_unit'] = df.groupby(['num', 'date'])['dosage_day_in_unit'].apply(list).values
    df_agg['dosage_unit'] = df.groupby(['num', 'date'])['dosage_unit'].apply(list).values
    df_agg.to_csv(os.path.join(df_filtered_path, 'DM_prescription.csv'))
    # ### Split data depending on data type
    # #### Integrated dataframe
    filtered_data_list = os.listdir(df_filtered_path)
    lab_list = []
    for f in filtered_data_list:
        if f.startswith('DM_lab'):
            lab_list.append(f)
    vital_list = ['DM_vital_BT.csv', 'DM_vital_DBP.csv', 'DM_vital_SBP.csv', 'DM_vital_PR.csv']
    demographic_list = ['DM_height.csv', 'DM_weight.csv']
    target_list = lab_list + vital_list + demographic_list + ['DM_prescription.csv']
    target_list = list(set(target_list) - set([
        'DM_lab_24_microalbumin_cr_ratio.csv',
        'DM_lab_eGFR_cal.csv',
        'DM_lab_eGFR_CKD-EPI_cal.csv',
        'DM_lab_Creatinine.csv',
        'DM_lab_Microalbumin.csv',
        'DM_lab_Protein.csv',
        'DM_lab_30_cal_Protein_cr_ratio.csv',
        'DM_lab_C_peptide_Basal.csv'
    ]))
    df = pd.read_csv(os.path.join(df_filtered_path, target_list[0]), index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    for f in target_list[1:]:
        print(f)
        f_df = pd.read_csv(os.path.join(df_filtered_path, f), index_col=0)
        f_df['date'] = pd.to_datetime(f_df['date'])
        df = pd.merge(df, f_df, on=['num', 'date'], how='outer')
    df_demo = pd.read_csv(os.path.join(df_filtered_path, 'DM_demographic.csv'))
    df = pd.merge(df, df_demo, on=['num'], how='left')
    df = df[~pd.isnull(df['indDate'])]
    df['date'] = pd.to_datetime(df['date']).astype(str)
    df['indDate'] = pd.to_datetime(df['indDate']).astype(str)
    df['year_gap'] = df['date'].str.split('-', expand=True)[0].astype(int) - df['indDate'].str.split('-', expand=True)[0].astype(int)
    df['age'] = df['age'] + df['year_gap']
    df['BMI'] = df['weight'] / (df['height'] * df['height']) * 10000
    if reject_data_before_diagnosis:
        df = df[(pd.to_datetime(df['date']) - pd.to_datetime(df['indDate'])).dt.days >= 0]
    for k in df.keys():
        if k.startswith('Unnamed:'):
            df.drop([k], axis=1, inplace=True)
    df.sort_values(by=['num', 'date'], ignore_index=True, inplace=True)
    df.to_csv(os.path.join(df_filtered_path, 'DM_all.csv'))

if __name__ == '__main__':
    main()
