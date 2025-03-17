import os
import requests
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import gc
import random

debug = True

class AndroZoo(object):
    def __init__(self, 
                 api_key, 
                 config, 
                 output_dir,
                 chunksize=100000, 
                 target_count=10000, 
                 num_threads=200, 
                 csv_path="latest.csv"):

        self.filtered_conditions = ""
        self.api_key = api_key
        self.config = config
        self.output_dir = output_dir
        self.chunksize = chunksize
        self.target_count = target_count
        self.num_threads = num_threads
        self.csv_path = os.path.join(output_dir, csv_path)


    def debug_print(self, message):
        if debug:
            print(message)

    # read csv by chunk
    def _czc_read_csv(self, parse_dates=['dex_date']):

        self.debug_print(f'start reading csv from：{self.csv_path}')
        chunks = []
        for chunk in tqdm(pd.read_csv(self.csv_path, parse_dates=parse_dates, chunksize=self.chunksize)):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

        return df


    # filter out apks 
    def czc_filter_apk(self):
        self.debug_print('start filtering APK')
        start_year_filter = int(self.config['start_year'])
        end_year_filter = int(self.config['start_year'])
        dex_size_limit = int(self.config['dex_size_limit'])
        apk_size_limit = int(self.config['apk_size_limit'])
        self.filtered_conditions = f"start_year_{start_year_filter}_end_year_{end_year_filter}_dex_size_{dex_size_limit}_apk_size_{apk_size_limit}"

        self.debug_print('reading csv')
        df = self._czc_read_csv(parse_dates=['dex_date'])
        df.set_index('dex_date', inplace=True)

        self.debug_print('filtering APK ...')
        goodware_filtered_df = df.loc[(df.index.year >= start_year_filter) & (df.index.year <= end_year_filter) &
                         (df['vt_detection'] == 0) & (df['dex_size'] < dex_size_limit) &
                         (df['apk_size'] < apk_size_limit)]
        goodware_sha256_list = goodware_filtered_df['sha256'].tolist()

        
        goodware_filtered_file = os.path.join(self.output_dir,f'goodware_filterd_{self.filtered_conditions}.txt')
        print(type(df.index.year))
        malware_filtered_df = df.loc[(df.index.year >= start_year_filter) & (df.index.year <= end_year_filter) &
                         (df['vt_detection'] >= 2) & (df['dex_size'] < dex_size_limit) &
                         (df['apk_size'] < apk_size_limit)]
        malware_sha256_list = malware_filtered_df['sha256'].tolist()

        malware_filtered_file = os.path.join(self.output_dir,f'malware_filterd_{self.filtered_conditions}.txt')
        
        self.debug_print('finished filtering!')

        self.debug_print('saving SHA256 list')
        with open(goodware_filtered_file, 'w') as f:
            for sha in goodware_sha256_list:
                f.write(sha + '\n')
        with open(malware_filtered_file, 'w') as f:
            for sha in malware_sha256_list:
                f.write(sha + '\n')
        self.debug_print('finished, all good')

        del df 
        gc.collect() 

        return goodware_filtered_file, malware_filtered_file


    # download filtered apks with multithreads
    def czc_download_apk_multithreaded(self):
        self.debug_print('start downloading ... ')
        goodware_downloaded_file = os.path.join(self.output_dir, f'goodware_downloaded_{self.filtered_conditions}.txt')
        malware_downloaded_file = os.path.join(self.output_dir, f'malware_downloaded_{self.filtered_conditions}.txt')

        if not os.path.exists(goodware_downloaded_file):
            open(goodware_downloaded_file, 'w').close()
        if not os.path.exists(malware_downloaded_file):
            open(malware_downloaded_file, 'w').close()

        goodware_filtered_file = os.path.join(self.output_dir,f'goodware_filterd_{self.filtered_conditions}.txt')
        malware_filtered_file = os.path.join(self.output_dir,f'malware_filterd_{self.filtered_conditions}.txt')
        
        with open(goodware_filtered_file, 'r') as f:
            goodware_sha256_list = f.readlines()
        with open(malware_filtered_file, 'r') as f:
            malware_sha256_list = f.readlines()

        with open(goodware_downloaded_file, 'r') as f:
            goodware_downloaded_list = f.readlines()
        with open(malware_downloaded_file, 'r') as f:
            malware_downloaded_list = f.readlines()

        goodware_sha256_list = [sha.strip() for sha in goodware_sha256_list]
        goodware_downloaded_list = [sha.strip() for sha in goodware_downloaded_list]
        malware_sha256_list = [sha.strip() for sha in malware_sha256_list]
        malware_downloaded_list = [sha.strip() for sha in malware_downloaded_list]

        goodware_to_download = list(set(goodware_sha256_list) - set(goodware_downloaded_list))
        random.shuffle(goodware_to_download)
        malware_to_download = list(set(malware_sha256_list) - set(malware_downloaded_list))
        random.shuffle(malware_to_download)

        goodware_download_dir = os.path.join(self.output_dir, 'goodware')
        os.makedirs(goodware_download_dir, exist_ok=True)
        malware_download_dir = os.path.join(self.output_dir, 'malware')
        os.makedirs(malware_download_dir, exist_ok=True)

        def download_task(sha256, downloaded_file, download_dir, pbar):
            url = f"https://androzoo.uni.lu/api/download?apikey={self.api_key}&sha256={sha256}"
            self.debug_print(f'downloading APK：{sha256}')
            try:
                response = requests.get(url, verify=True, timeout=10)
                if response.status_code == 200:
                    apk_name = self.config['start_year']+'_'+sha256 + '.apk'
                    with open(os.path.join(download_dir, apk_name), 'wb') as file:
                        file.write(response.content)
                    with open(downloaded_file, 'a') as f:
                        f.write(sha256 + '\n')
                    pbar.update(1)
                    self.debug_print(f'SUCCESS: {sha256}')
                else:
                    self.debug_print(f'FAIL: {sha256}, CODE: {response.status_code}')
            except Exception as e:
                self.debug_print(f'FAIL: {sha256}, CODE: {e}')

        with tqdm(total=self.target_count, desc='downloading goodwares ....') as pbar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                while len(goodware_downloaded_list) < self.target_count and goodware_to_download:
                    sha256 = goodware_to_download.pop()
                    future = executor.submit(download_task, sha256, goodware_downloaded_file, goodware_download_dir, pbar)
                    futures.append(future)
                    goodware_downloaded_list.append(sha256)
                for future in futures:
                    future.result()
        self.debug_print('Finished Downloading ...')

        with tqdm(total=self.target_count, desc='downloading malwares ....') as pbar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                while len(malware_downloaded_list) < self.target_count and malware_to_download:
                    sha256 = malware_to_download.pop()
                    future = executor.submit(download_task, sha256, malware_downloaded_file, malware_download_dir, pbar)
                    futures.append(future)
                    malware_downloaded_list.append(sha256)
                for future in futures:
                    future.result()
        self.debug_print('Finished Downloading ...')
