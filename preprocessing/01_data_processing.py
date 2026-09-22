import os
import sys

import numpy as np
import pandas as pd

from paths import RAW_ROOT

def snubh_exam(df, exam_code, df_path, f):
    unique_exam_code = list(set(df['검사코드'].tolist()))

    if len(unique_exam_code) > 1:
        for code in unique_exam_code:
            try:
                code_list = exam_code[exam_code['코드1'] == code][[
                    '코드1', '코드2', '코드3', '코드4', '코드5'
                ]].values
                if all([i in code_list for i in unique_exam_code]):
                    exam_name = exam_code[exam_code['코드1'] ==
                                          code]['검사명'].iloc[0]
                    df.rename(columns={
                        '검사시행일': 'date',
                        '검사시행일자시간': 'date',
                        '검사결과': exam_name
                    },
                              inplace=True)
                    df.drop(['검사코드', '검사명', '검사세부항목명'],
                            axis=1,
                            inplace=True)
                    break

            except KeyError:
                pass
    else:
        exam_name = exam_code[exam_code['코드1'] == unique_exam_code[0]]['검사명'].iloc[0]
        df.rename(columns={
            '검사시행일': 'date',
            '검사결과': exam_name
        },
                  inplace=True)
        df.drop(['검사코드', '검사명', '검사세부항목명'], axis=1, inplace=True)

    if len(df.keys()) == 2:
        df.to_csv(os.path.join(df_path, f.replace('검사결과', 'lab').replace('계산값', 'cal')))
        print('Processed exam results of\t' + f + '\tto ' + f.replace('검사결과', 'lab').replace('계산값', 'cal'))
        print('Exam name:\t' + exam_name)
        
def vital(df, df_path, df_name):
    if len(df['항목명'].unique()) > 1:
        df['항목명'] = df['항목명'].str.extract(r'([a-zA-Z]*)')
    df.rename(columns={
        '작성일자': 'date',
        '항목값': df['항목명'].iloc[0]
    },
              inplace=True)
    df.drop(['항목명'], axis=1, inplace=True)
    df.to_csv(os.path.join(df_path, df_name))


def main():
    data_path = RAW_ROOT
    hospital = 'snubh'
    session = '2'
    base_path = os.path.join(data_path, hospital, session)
    raw_data_path = os.path.join(base_path, 'DM_csv')
    df_path = os.path.join(base_path, 'DM_df')
    os.makedirs(df_path, exist_ok=True)
    data_list = os.listdir(raw_data_path)
    # convert filenames from Korean to English
    for f in os.listdir(raw_data_path):
    # load code information
        if os.path.isdir(os.path.join(raw_data_path, f)):
            continue
        if f[-3:] == 'csv':
            df = pd.read_csv(os.path.join(raw_data_path, f))
            new_f = f.split('_')
            if '검사명' in df.keys():
                new_f[1] = 'lab'
            elif '항목명' in df.keys():
                if df['항목명'].iloc[0] in ['SBP', 'DBP', 'BT', 'PR']:
                    new_f[1] = 'vital'
                elif (df['항목명'].iloc[0] == '몸무게') | (df['항목명'].iloc[0] == '체중'):
                    new_f[1] = 'weight'
                elif df['항목명'].iloc[0] == '키':
                    new_f[1] = 'height'
            elif '약품처방일' in df.keys():
                new_f[1] = 'prescription'
            elif 'indDate' in df.keys():
                new_f[1] = 'demographic'
            elif '진단일자' in df.keys():
                new_f[1] = 'diagnosis'
            elif '수진일당시흡연여부' in df.keys():
                new_f[1] = 'smokingdrinking'
            elif '입원일자' in df.keys():
                new_f[1] = 'admission'
            else:
                print(f)
                continue

            new_f = '_'.join(new_f)
            if new_f[-3:] != 'csv':
                new_f += '.csv'
            os.rename(os.path.join(raw_data_path, f), os.path.join(raw_data_path, new_f))
            print(f'rename {f:>20} to {new_f}')
    data_list = os.listdir(raw_data_path)
    # SNUBH
    exam_code = pd.read_excel(os.path.join(data_path, 'SNUBH_DMCKD_ref.xlsx'),
    # ## Rename columns
    # convert column names
    # ### Urine tests
                              sheet_name='검사코드')
    decoding_failure_list = []
    os.makedirs(df_path, exist_ok=True)
    for f in data_list:
        if os.path.isdir(os.path.join(raw_data_path, f)):
            continue
        if f[-3:] == 'csv':
    # ### Demographics
            try:
                df = pd.read_csv(os.path.join(raw_data_path, f),
                                 index_col=0,
                                 encoding=sys.getfilesystemencoding())
            except UnicodeDecodeError:
                decoding_failure_list.append(f)

            if '검사코드' in df.keys():
                snubh_exam(df, exam_code, df_path, f)
            if '항목명' in df.keys():
                vital(df, df_path, f.replace('생체징후', 'vital'))
                print(f'Processed biomarkers of {f :<20} to ' + f.replace('생체징후', 'vital'))
    print(decoding_failure_list)
    try:
        os.rename(os.path.join(df_path, 'DM_몸무게.csv'), os.path.join(df_path, 'DM_weight.csv'))
        os.rename(os.path.join(df_path, 'DM_키.csv'), os.path.join(df_path, 'DM_height.csv'))
    except FileNotFoundError:
        pass
    if session == '1':
        df = pd.read_csv(os.path.join(raw_data_path, 'DM_lab_Urine.csv'))
        df.rename(columns={'검사시행일자시간': 'date'}, inplace=True)
        exam_list = df['검사명'].unique()

        for exam in exam_list:
            df_exam = df[df['검사명'] == exam]
            df_exam.rename(columns={'검사결과': exam}, inplace=True)
            df_exam.drop(['검사명', '검사코드'], axis=1, inplace=True)
            exam_name = exam.split(' ')[0]
            df_exam.to_csv(os.path.join(df_path, 'DM_lab_' + exam_name + '.csv'))
    try:
        os.rename(
            os.path.join(raw_data_path, 'DM_성별나이기준일자.csv'), 
            os.path.join(raw_data_path, 'DM_demographic.csv')
        )
    except FileNotFoundError:
        pass
    df = pd.read_csv(os.path.join(raw_data_path, 'DM_demographic.csv'), index_col=0)
    df.rename(columns={'성별': 'sex'}, inplace=True)
    df['sex'].replace(to_replace='여성', value=0, inplace=True)
    df['sex'].replace(to_replace='남성', value=1, inplace=True)
    df['sex'].replace(to_replace='F', value=0, inplace=True)
    df['sex'].replace(to_replace='M', value=1, inplace=True)
    df.to_csv(os.path.join(df_path, 'DM_demographic.csv'))
    # ### Prescriptions
    for f in os.listdir(raw_data_path):
        if f.startswith('DM_모든처방자료'):
            os.rename(
                os.path.join(raw_data_path, f), 
                os.path.join(raw_data_path, f.replace('모든처방자료', 'prescription'))
            )
    df_list = []
    for f in os.listdir(raw_data_path):
        if f.startswith('DM_prescription'):
            df_list.append(f)
    df = pd.DataFrame()
    for f in df_list:
        df = pd.concat([
            df,
            pd.read_csv(os.path.join(raw_data_path, f), index_col=0)
        ], axis=0)
    df.fillna(value=np.nan, inplace=True)
    df.rename(columns={
        '약품처방일': 'date',
        '원내약품코드': 'code',
        '약품명(상품명)': 'commercial_name',
        '약품명(성분명)': 'ingredient',
        '포장단위1일약품투여량': 'dosage_day_in_pack',
        '포장단위1회약품투여량': 'dosage_time_in_pack',
        '처방단위': 'unit',
        '1일처방량': 'dosage_day',
        '1회처방량': 'dosage_time',
        '투약일수': 'dosage_duration',
        '약품명(일반명)': 'name'
    }, inplace=True)
    original_num = df.shape[0]
    df = df[~df['commercial_name'].str.contains('임상', na=False)]
    df = df[~df['ingredient'].str.contains('임상', na=False)]
    df = df[~df['name'].str.contains('임상', na=False)]
    print(f'{original_num - df.shape[0]} data have been removed')
    df['dosage_pack_unit'] = df['dosage_time_in_pack'].str.extract(r"\/(.*?)\)")
    df['dosage_unit'] = df['dosage_time_in_pack'].str.extract(r"\(.*?([a-zA-Z%+]+)\/")
    df['dosage_pack'] = df['dosage_time_in_pack'].str.extract(r"\(([0-9.]+).*?\/")
    df['dosage_pack'] = df['dosage_pack'].astype(float)
    df['dosage_day_in_unit'] = df['dosage_day_in_pack'] * df['dosage_pack']
    df = df.reset_index()
    df = df[['num', 'date', 'code', 'ingredient', 'dosage_day_in_unit', 'dosage_unit']]
    df.to_csv(os.path.join(df_path, 'DM_prescription.csv'))
    # ### eGFR
    try:
    # ## microalbumin_cr_ratio
        os.rename(os.path.join(raw_data_path, 'DM_검사결과_13_eGFR.csv'), os.path.join(raw_data_path, 'DM_lab_13_eGFR.csv'))
        os.rename(os.path.join(raw_data_path, 'DM_검사결과_13_eGFR_변환결과.csv'), 
                  os.path.join(raw_data_path, 'DM_lab_13_eGFR_cal.csv'))
        os.rename(os.path.join(raw_data_path, 'DM_검사결과_13_eGFR_CKD-EPI_변환결과.csv'),
                  os.path.join(raw_data_path, 'DM_lab_13_eGFR_CKD-EPI_cal.csv'))

    except FileNotFoundError:
        pass
    try:
        df_cal_m = pd.read_csv(os.path.join(raw_data_path, 'DM_lab_13_eGFR_cal.csv'))
        df_cal_e = pd.read_csv(os.path.join(raw_data_path, 'DM_lab_13_eGFR_CKD-EPI_cal.csv'))
        df = pd.read_csv(os.path.join(raw_data_path, 'DM_lab_13_eGFR.csv'))
        filename = 'DM_lab_13_eGFR.csv'
    except FileNotFoundError:
        df_cal_m = pd.read_csv(os.path.join(df_path, 'DM_lab_eGFR_cal.csv')).drop_duplicates(['num', 'date'])
        df_cal_m.columns = ['num', '검사시행일', '검사결과']
        df_cal_e = pd.read_csv(os.path.join(df_path, 'DM_lab_eGFR_CKD-EPI_cal.csv')).drop_duplicates(['num', 'date'])
        df_cal_e.columns = ['num', '검사시행일', '검사결과']
        df = df_cal_e.copy()
        filename = 'DM_lab_eGFR.csv'
    df['eGFR'] = df['검사결과'].str.extract(r'([\.0-9]*)').apply(pd.to_numeric, errors='coerce').values
    print(f'Number of original data is {df.shape[0]}')
    df_cal_e.rename(columns={'검사결과': 'eGFRe'}, inplace=True)
    df_cal_m.rename(columns={'검사결과': 'eGFRm'}, inplace=True)
    df = df.merge(df_cal_e[['num', '검사시행일', 'eGFRe']], on=['num', '검사시행일'], how='outer')
    print(f'Number of data after eGFR-E integration is {df.shape[0]}')
    df = df.merge(df_cal_m[['num', '검사시행일', 'eGFRm']], on=['num', '검사시행일'], how='outer')
    print(f'Number of data after eGFR-M integration is {df.shape[0]}')
    df['eGFR'] = np.where(pd.isnull(df['eGFR']), df['eGFRe'], df['eGFR'])
    df['eGFR'] = np.where(pd.isnull(df['eGFR']), df['eGFRm'], df['eGFR'])
    df_egfr = df[['num', '검사시행일', 'eGFR']]
    df_egfr.columns=['num', 'date', 'eGFR']
    df_egfr.dropna(subset=['eGFR'], inplace=True)
    df_egfr.to_csv(os.path.join(df_path, filename))
    df = pd.read_csv(os.path.join(df_path, 'DM_lab_24_microalbumin_cr_ratio.csv'))
    df_cal = pd.read_csv(os.path.join(df_path, 'DM_lab_24_cal_microalbumin_cr_ratio.csv'))
    df_cal.rename(columns={'microalbumin_cr_ratio': 'cal_microalbumin_cr_ratio', 'lab_dt': 'date'}, inplace=True)
    df = df.merge(df_cal, on=['num', 'date'], how='outer')
    df['microalbumin_cr_ratio'] = np.where(pd.isnull(df['microalbumin_cr_ratio']), df['cal_microalbumin_cr_ratio'], df['microalbumin_cr_ratio'])
    df = df[['num', 'date', 'microalbumin_cr_ratio']]
    os.remove(os.path.join(df_path, 'DM_lab_24_cal_microalbumin_cr_ratio.csv'))
    df.to_csv(os.path.join(df_path, 'DM_lab_24_microalbumin_cr_ratio.csv'))

if __name__ == '__main__':
    main()
