import pandas as pd

def carregar_dados():
    dicom_data = pd.read_csv("cbis_ddsm_dataset/csv/dicom_info.csv")
    image_dir = 'cbis_ddsm_dataset/jpeg'

    full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
    cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
    roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

    full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    roi_mask_images = roi_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    full_mammogram_images.iloc[0]

    full_mammogram_dict = dict()
    cropped_dict = dict()
    roi_mask_dict = dict()

    for dicom in full_mammogram_images:
        # print(dicom)
        key = dicom.split("/")[2]
        # print(key)
        full_mammogram_dict[key] = dicom
    for dicom in cropped_images:
        key = dicom.split("/")[2]
        cropped_dict[key] = dicom
    for dicom in roi_mask_images:
        key = dicom.split("/")[2]
        roi_mask_dict[key] = dicom


    mass_train_data = pd.read_csv('cbis_ddsm_dataset/csv/mass_case_description_train_set.csv')
    mass_test_data = pd.read_csv('cbis_ddsm_dataset/csv/mass_case_description_test_set.csv')
    calc_train_data = pd.read_csv('cbis_ddsm_dataset/csv/calc_case_description_train_set.csv')
    calc_test_data = pd.read_csv('cbis_ddsm_dataset/csv/calc_case_description_test_set.csv')

    def fix_image_path_mass(dataset):
        for i, img in enumerate(dataset.values):
            img_name = img[11].split("/")[2]
            if img_name in full_mammogram_dict:
                dataset.iloc[i, 11] = full_mammogram_dict[img_name]

            img_name = img[12].split("/")[2]
            if img_name in cropped_dict:
                dataset.iloc[i, 12] = cropped_dict[img_name]
            
            img_name = img[13].split("/")[2]
            if img_name in roi_mask_dict:
                dataset.iloc[i, 13] = roi_mask_dict[img_name]


    fix_image_path_mass(mass_train_data)
    fix_image_path_mass(mass_test_data)

    def fix_image_path_calc(dataset):
        for i, img in enumerate(dataset.values):
            img_name = img[11].split("/")[2]
            if img_name in full_mammogram_dict:
                dataset.iloc[i, 11] = full_mammogram_dict[img_name]

            img_name = img[12].split("/")[2]
            if img_name in cropped_dict:
                dataset.iloc[i, 12] = cropped_dict[img_name]
            
            img_name = img[13].split("/")[2]
            if img_name in roi_mask_dict:
                dataset.iloc[i, 13] = roi_mask_dict[img_name]

    fix_image_path_calc(calc_train_data)
    fix_image_path_mass(calc_test_data)


    df_train = pd.concat([mass_train_data, calc_train_data], ignore_index=True)

    df_train['pathology_binary'] = df_train['pathology'].apply(lambda x: 1 if x == 'MALIGNANT' else 0)
    df_train['pathology_binary'] = df_train['pathology_binary'].astype(str)

    df_test = pd.concat([mass_test_data, calc_test_data], ignore_index=True)
    df_test['pathology_binary'] = df_test['pathology'].apply(lambda x: 1 if x == 'MALIGNANT' else 0)
    df_test['pathology_binary'] = df_test['pathology_binary'].astype(str)

    return df_train, df_test 