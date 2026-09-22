import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (DATA_TYPE, MIN_BASELINE_EGFR, MIN_EGFR_MEASUREMENTS,
                    OBSERVATION_WINDOWS, PREDICTION_INTERVALS)
from paths import RAW_ROOT

    

def preproc(dataframe):
    dataframe.sort_values(by=['num', 'date'], inplace=True, ignore_index=True)
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['indDate'] = pd.to_datetime(dataframe['indDate'])
    dataframe['relative_date_in_month'] = (dataframe['date'] - dataframe['indDate']) / np.timedelta64(1, 'M')
    dataframe['relative_date_in_month'] = dataframe['relative_date_in_month'].astype(int)
    dataframe['relative_date_in_day'] = (dataframe['date'] - dataframe['indDate']) / np.timedelta64(1, 'D')
    dataframe['relative_date_in_day'] = dataframe['relative_date_in_day'].astype(int)

    # integrate C-peptide and C-peptide basal
    if 'C_peptide_Basal' in dataframe.keys():
        dataframe['C_peptide'] = np.where(pd.isnull(dataframe['C_peptide']), dataframe['C_peptide_Basal'], dataframe['C_peptide'])
    if ('cal_microalbumin_cr_ratio' in dataframe.keys()):
        if ('microalbumin_cr_ratio' in dataframe.keys()):
            dataframe.drop('microalbumin_cr_ratio', axis=1, inplace=True)
        dataframe.rename(columns={'cal_microalbumin_cr_ratio': 'microalbumin_cr_ratio'}, inplace=True)
    
    if 'new_eGFR' in dataframe.keys():
        dataframe.rename(columns={'new_eGFR': 'eGFR'}, inplace=True)
    return dataframe.reindex(sorted(dataframe.columns), axis=1)

def main():
    data_path = RAW_ROOT
    hospital = 'snubh'
    sessions = ['1', '2']
    base_path = os.path.join(data_path, hospital)
    df_filtered_path = os.path.join(base_path, 'DM_df_filtered')
    df_segmented_path = os.path.join(base_path, 'DM_df_segmented', DATA_TYPE)
    os.makedirs(df_segmented_path, exist_ok=True)
    df = pd.DataFrame()
    for session in sessions:
        base_path = os.path.join(data_path, hospital, session)
        df_filtered_path = os.path.join(base_path, 'DM_df_filtered')
        if len(df) == 0:
            df = pd.read_csv(os.path.join(df_filtered_path, 'DM_all.csv'), index_col=0)
            df = preproc(df)
        else:
            tmp = pd.read_csv(os.path.join(df_filtered_path, 'DM_all.csv'), index_col=0)
            tmp = preproc(tmp)
            assert (df.keys() == tmp.keys()).all()
            df = pd.concat((df, tmp), axis=0)
    # A small number of individual records were dropped here as implausible
    # (transcription errors in the source export). The list identified them by
    # patient id and visit date, which is individual-level data, so it is not
    # published with this code. Reproducing the exact cohort therefore needs
    # that list from the data custodian; without it this step is a no-op and a
    # handful of anomalous records remain, which the quantile filters below
    # largely catch anyway.
    # drop values outside the 0.1st-99.9th percentile of each variable
    df = df[~((df['Glucose']<df['Glucose'].quantile(0.001))&(df['Glucose']>df['Glucose'].quantile(0.999)))]
    df = df[~((df['Potassium']<df['Potassium'].quantile(0.001))&(df['Potassium']>df['Potassium'].quantile(0.999)))]
    df = df[~((df['Phosphorus']<df['Phosphorus'].quantile(0.001))&(df['Phosphorus']>df['Phosphorus'].quantile(0.999)))]
    df = df[~((df['eGFR']<df['eGFR'].quantile(0.001))&(df['eGFR']>df['eGFR'].quantile(0.999)))]
    df = df[~(df['microalbumin_cr_ratio']>df['microalbumin_cr_ratio'].quantile(0.999))]
    df = df[~(df['Protein_cr_ratio']>df['Protein_cr_ratio'].quantile(0.999))]
    data_keys = ['num', 'date', 'Albumin', 'Bilirubin', 'BUN', 'C_peptide',
               'Calcium', 'Chloride', 'Cholesterol', 'CO2_total', 
               'Creatinine_serum', 'creatinine_urine', 'CRP', 'eGFR', 'FBS', 'GGT',
               'Glucose', 'GOT', 'GPT', 'Hb', 'HbA1c', 'Hct', 'HDL', 'LDL',
               'microalbumin_cr_ratio', 'microalbumin_urine', 'Phosphorus', 'PLT', 'Potassium',
               'Protein_cr_ratio', 'Protein_total', 'Protein_urine', 
               'Sodium', 'Triglyceride', 'Troponin_i', 'Uric_Acid', 'WBC',
               'UA_Albumin', 'BT', 'DBP', 'SBP', 'PR', 'height', 'weight', 'code',
               'ingredient', 'dosage_day_in_unit', 'dosage_unit', 'sex', 'age',
               'indDate', 'year_gap', 'BMI', 'relative_date_in_month',
               'relative_date_in_day']
    inputs = []
    targets = []
    times = []
    df.drop_duplicates(inplace=True)
    for dur in OBSERVATION_WINDOWS:
        for intv in PREDICTION_INTERVALS:
            print('Duration {}, Interval {}'.format(dur, intv))
            df_egfr = df[df['eGFR'] > 0]
            df_egfr.drop_duplicates(['num', 'relative_date_in_month'], keep='last', inplace=True)

            target_df_egfr = df_egfr[['num', 'relative_date_in_month', 'eGFR']]
            target_df_egfr.rename(columns={'eGFR': 'target_eGFR', 'relative_date_in_month': 'target_month'}, inplace=True)
            target_df_egfr['end_observation_month'] = target_df_egfr['target_month'] - intv

            target_df_egfr['CKD_label'] = np.where(target_df_egfr['target_eGFR'] < 60, 1, 0)
            target_df_egfr['CKD_label'] += np.where(target_df_egfr['target_eGFR'] < 30, 1, 0)
            target_df_egfr['CKD_label'] += np.where(target_df_egfr['target_eGFR'] < 15, 1, 0)

            df_egfr = pd.merge(df_egfr, target_df_egfr, 
                               left_on=['num', 'relative_date_in_month'], 
                               right_on=['num', 'end_observation_month'],
                               how='inner')

            for i in tqdm(range(df_egfr.shape[0])):
                duration = list(range(df_egfr['end_observation_month'].iloc[i] - dur + 1, df_egfr['end_observation_month'].iloc[i] + 1))
                current_df = df[df['num'] == df_egfr['num'].iloc[i]]
                current_df = current_df[current_df['relative_date_in_month'].isin(duration)]
                current_duration = df_egfr['end_observation_month'].iloc[i] - current_df['relative_date_in_month'].min()

                # reject windows holding six months or less of actual observation
                if (dur == 12) & (current_duration <= 6):
                    continue

                # reject cases with too few eGFR measurements during the observation
                if current_df['eGFR'].notna().sum() < MIN_EGFR_MEASUREMENTS:
                    continue

                # reject cases where the patients already have kidney failure during the observation
                if np.nanmin(current_df['eGFR']) < MIN_BASELINE_EGFR:
                    continue

                inputs.append(current_df[data_keys].values)
                targets.append(df_egfr[['num', 'date', 'relative_date_in_month', 'eGFR', 'target_month', 'target_eGFR', 'CKD_label']].iloc[i].values)
                times.append([dur, intv])
    # convert date to string
    str_convert_target = [data_keys.index('date')] + [data_keys.index('indDate')]
    new_inputs = []
    for data in inputs:
        new_data = data.copy()
        for seq in new_data:
            seq[str_convert_target[0]] = str(seq[str_convert_target[0]])
            seq[str_convert_target[1]] = str(seq[str_convert_target[1]])
        new_inputs.append(new_data)
    new_inputs = np.array(new_inputs, dtype='object')
    del inputs
    inputs = new_inputs
    targets = np.array(targets)
    times = np.array(times)
    # save
    joblib.dump({
                 'inputs': inputs,
                 'targets': targets,
                 'times': times
                }, os.path.join(df_segmented_path, 'segmented_all.pkl'))
    # check class distribution
    CKD_target = targets[:, -1]
    print('Class normal: {}'.format(sum(CKD_target==0)))
    print('Class CKD 3: {}'.format(sum(CKD_target==1)))
    print('Class CKD 4: {}'.format(sum(CKD_target==2)))
    print('Class CKD 5: {}'.format(sum(CKD_target==3)))
    # ### Split into categories
    category = {
        'basic': ['sex', 'age', 'height', 'weight', 'BMI', 'SBP', 'DBP', 'BT', 'PR'],
        'CBC_panel': ['WBC', 'Hb', 'Hct', 'PLT'],
        'admission_panel': [
            'Sodium',
            'Potassium',
            'Chloride',
            'CO2_total',
            'Calcium',
            'Phosphorus',
            'BUN',
            'Creatinine_serum',
            'eGFR',
            'Uric_Acid',
            'Bilirubin',
            'Albumin',
            'Cholesterol',
            'GOT',
            'GPT',
            'GGT',
            'Glucose'
        ],
        'inflammation_marker': ['CRP'],
        'DM_control': ['HbA1c', 'FBS'],
        'insulin': ['C_peptide'],
        'dyslipidemia_marker': ['HDL', 'LDL', 'Triglyceride'],
        'urine_marker': ['Protein_urine', 'RBC_urine', 'UA_Albumin'] if 'RBC_urine' in data_keys else ['Protein_urine', 'UA_Albumin'],
        'proteinuria_marker': ['microalbumin_urine', 'microalbumin_cr_ratio', 'Protein_cr_ratio', 'Protein_total', 'creatinine_urine'],
        'renal_panel': ['Sodium', 'Potassium', 'Chloride', 'CO2_total', 'Calcium', 'Phosphorus', 'BUN', 'Creatinine_serum', 'eGFR'],
        'co2': ['pCO2'] if 'pCO2' in data_keys else [],
        'cardiac_marker': ['Troponin_i']
    }
    meta_keys = ['num', 'date', 'indDate', 'year_gap', 'relative_date_in_month', 'relative_date_in_day']
    prs_keys = ['code', 'ingredient', 'dosage_day_in_unit', 'dosage_unit']
    info_keys = ['sex', 'age']
    meta_indices = [data_keys.index(k) for k in meta_keys]
    info_indices = [data_keys.index(k) for k in info_keys]
    # #### prescription
    divided_input = []
    target_index = [data_keys.index(k) for k in prs_keys]
    for data in inputs:
        current_data = data[:, target_index]
        null_index = pd.isnull(current_data).all(-1)

        if sum(null_index) < len(current_data):
            current_data = data[~null_index][:, meta_indices + target_index]
        else:
            current_data = data[0, meta_indices + target_index]

        if current_data.ndim == 1:
            current_data = current_data.reshape(1, -1)

        if current_data.shape[1] != len(meta_indices) + len(target_index):
            raise ValueError(
                    'Segment has %d columns; expected %d meta + %d target columns'
                    % (current_data.shape[1], len(meta_indices), len(target_index)))

        divided_input.append(current_data)
    joblib.dump({
                'inputs': divided_input,
                'targets': targets,
                'times': times
                }, os.path.join(df_segmented_path, 'segmented_prescription.pkl'))
    # #### divide based on frequency
    # reload data
    data = joblib.load(os.path.join(df_segmented_path, 'segmented_all.pkl'))
    inputs = data['inputs']
    targets = data['targets']
    times = data['times']
    usual_category = set(
                    category['basic'] + \
                    category['CBC_panel'] + \
                    category['admission_panel'] + \
                    category['DM_control'] + \
                    category['dyslipidemia_marker'] + \
                    category['urine_marker'] + \
                    category['proteinuria_marker'] + \
                    category['renal_panel']
                    )
    unusual_category = set(
                    category['inflammation_marker'] + \
                    category['insulin'] + \
                    category['co2'] + \
                    category['cardiac_marker']
                    )
    usual_category = sorted(list(usual_category))
    unusual_category = sorted(list(unusual_category))
    frequency_based_category = {'usual': usual_category,
                                'unusual': unusual_category}
    # obtain target indices
    frequency_based_category_indices = {}
    for k, v in frequency_based_category.items():
        index = [data_keys.index(vi) for vi in v]
        frequency_based_category_indices[k] = index
    divided_inputs = {}
    input_keys = {}
    for k in frequency_based_category.keys():
        print(k)
        if k == 'usual':
            target_index = list(set(frequency_based_category_indices[k]) - set(info_indices)) # to exclude columns of sex and age from the null condition
        else:
            target_index = frequency_based_category_indices[k]

        divided_input = []
        for data in inputs:
            current_data = data[:, target_index]
            null_index = pd.isnull(current_data).all(-1)

            if k == 'usual':
                target_index = frequency_based_category_indices[k]

            if sum(null_index) < len(current_data):
                current_data = data[~null_index][:,meta_indices + target_index]
            else:
                current_data = data[0, meta_indices + target_index]

            if current_data.ndim == 1:
                current_data = current_data.reshape(1, -1)

            if current_data.shape[1] != len(meta_indices) + len(target_index):
                raise ValueError(
                        'Segment has %d columns; expected %d meta + %d target columns'
                        % (current_data.shape[1], len(meta_indices), len(target_index)))

            divided_input.append(current_data)

        divided_inputs[k] = divided_input
        input_keys[k] = np.array(data_keys)[meta_indices + target_index]
    joblib.dump({
                 'inputs': divided_inputs,
                 'targets': targets,
                 'times': times,
                 'keys': input_keys
                }, os.path.join(df_segmented_path, 'segmented_frequency.pkl'))
    # ### key info
    # ##### indices and importance tokens
    os.path.exists(os.path.join(df_segmented_path, 'key_info.pkl'))
    os.listdir(os.path.join(df_segmented_path))
    if not os.path.exists(os.path.join(df_segmented_path, 'key_info.pkl')):
        drug_code = pd.read_csv(os.path.join(data_path, 'code_label_integ.csv'))
        drug_code = {drug['ingredient']: drug['importance'] for _, drug in drug_code.iterrows()}
        lab = {
         'Albumin': 1.,
         'Bilirubin': 0.,
         'BUN': 1.,
         'C_peptide': 0.,
         'Calcium': 0.,
         'Chloride': 0.,
         'Cholesterol': 1.,
         'CO2_total': 0.,
         'Creatinine_serum': 1.,
         'creatinine_urine': 1.,
         'CRP': 0.,
         'eGFR': 1.,
         'FBS': 1.,
         'GGT': 0.,
         'Glucose': 1.,
         'GOT': 0.,
         'GPT': 0.,
         'Hb': 0.,
         'HbA1c': 1.,
         'Hct': 0.,
         'HDL': 1.,
         'LDL': 1.,
         'microalbumin_cr_ratio': 1.,
         'microalbumin_urine': 1.,
         'pCO2': 0.,
         'Phosphorus': 0.,
         'PLT': 0.,
         'Potassium': 0.,
         'Protein_cr_ratio': 1.,
         'Protein_total': 0.,
         'Protein_urine': 1.,
         'RBC_urine': 0.,
         'Sodium': 0.,
         'Triglyceride': 0.,
         'Troponin_i': 0.,
         'Uric_Acid': 0.,
         'WBC': 0.,
         'UA_Albumin': 1.,
         'BT': 0.,
         'DBP': 1.,
         'SBP': 1.,
         'PR': 0.,
         'height': 0.,
         'weight': 1.,
         'sex': 1.,
         'age': 1.,
         'BMI': 1.
        }

        key_info = {
            'categories': category,
            'usual_categories': usual_category,
            'unusual_categories': unusual_category,
            'all_keys': data_keys,
            'meta_keys': meta_keys,
            'prs_keys': prs_keys,
            'importance_token': {'drug': drug_code, 'lab': lab},
        }
        joblib.dump(key_info, os.path.join(df_segmented_path, 'key_info.pkl'))

if __name__ == '__main__':
    main()
