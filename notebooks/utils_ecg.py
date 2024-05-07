import os
import pandas as pd
import wfdb
import glob
from tqdm import tqdm



def create_ecg_jpeg_pos_neg_subsets():
    import shutil
    import pandas as pd
    import os
    data_dir = '/data/vision/polina/scratch/barbaralam/cxrpe/data/Modeling'
    df_neg = pd.read_csv(os.path.join(data_dir, 'EKGneg48.csv'))
    df_pos = pd.read_csv(os.path.join(data_dir, 'EKGpos48.csv'))
    df = pd.concat([df_neg, df_pos])

    ekg_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs_plots'

    # ekg_ids = [x.split('/')[-1] for x in df_pos['path'].tolist()]
    # target_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs_plots_pos'

    ekg_ids = [x.split('/')[-1] for x in df_neg[:1675]['path'].tolist()]
    target_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs_plots_neg'


    os.makedirs(target_dir, exist_ok=True)

    for ekg_id in ekg_ids:
        source_file = os.path.join(ekg_dir, f'{ekg_id}.jpeg')
        target_file = os.path.join(target_dir, f'{ekg_id}.jpeg')
        if os.path.isfile(source_file):
            shutil.copy(source_file, target_file)



def save_ecg_url_to_file():
    """first save ekg urls to be downloaded to a file, 
        then use following `wget` to download them all at once.

        ```
        wget -i ecg_urls.txt -P /data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs/
        ```
    """
    data_dir = '/data/vision/polina/scratch/barbaralam/cxrpe/data/Modeling'
    df_neg = pd.read_csv(os.path.join(data_dir, 'EKGneg48.csv'))
    df_pos = pd.read_csv(os.path.join(data_dir, 'EKGpos48.csv'))
    df = pd.concat([df_neg, df_pos])

    paths = [f'https://physionet.org/files/mimic-iv-ecg/1.0/{p}' for p in df['path'].tolist()]

    with open('ecg_urls.txt', 'w') as f:
        for path in paths:
            f.write(f'{path}.hea\n')
            f.write(f'{path}.dat\n')



def convert_ecg_wfdb_to_jpeg(args):
    ecg_file, image_file = args

    try:
        record = wfdb.rdrecord(ecg_file)
        fig = wfdb.plot_wfdb(
            record=record, 
            title=' ', 
            return_fig=True,
            ecg_grids='all',
            figsize=(24,18))
        fig.savefig(image_file, dpi=300)
        fig.clear()
    except:
        print(f'ecg_file: {ecg_file} cannot be processed')



def convert_ecg_to_image(ecg_dir, output_dir):
    """Convert ecgs in `ecg_dir` to `jpeg` images and save them to `output_dir`.

        ```
        from utils_ecg import convert_ecg_to_image
        ecg_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs/'
        output_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs_plots/'
        convert_ecg_to_image(ecg_dir, output_dir)
        ```
    """
    from rosemary import joblib_parallel_process

    os.makedirs(output_dir, exist_ok=True)

    ecg_ids = sorted(list(set([x.split('.')[0] for x in os.listdir(ecg_dir)])))
    args = [(
        os.path.join(ecg_dir, ecg_id),
        os.path.join(output_dir, f'{ecg_id}.jpeg')
    ) for ecg_id in ecg_ids]

    joblib_parallel_process(
        fn=convert_ecg_wfdb_to_jpeg,
        iterable=args,
         n_jobs=30,
         use_tqdm=True)



if __name__ == '__main__':
    ecg_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs/'
    output_dir = '/data/vision/polina/scratch/wpq/github/cxrpe/data/ecgs_plots/'
    convert_ecg_to_image(ecg_dir, output_dir)