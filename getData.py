from AndroZoo import AndroZoo
import configparser



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    az = AndroZoo(api_key=str(config["access"]["APIKEY"]), 
                 config=config["config"], 
                 output_dir=str(config["path"]["data"]),
                 chunksize=int(config["setting"]["chunksize"]) if config["setting"]["chunksize"]!="" else 100000, 
                 target_count=int(config["setting"]["target_count"]) if config["setting"]["target_count"]!="" else 100000, 
                 num_threads=int(config["setting"]["num_thread"]) if config["setting"]["num_thread"]!="" else 200, 
                 csv_path=str(config["path"]["csv"]) if config["path"]["csv"]!="" else "latest.csv")

    filtered_file = az.czc_filter_apk()

    links_dir = az.czc_download_apk_multithreaded()
