"""
NOTICE: This script accesses the International Telecommunication Union (ITU)
website to download data files for the user's personal use.

see https://www.itu.int/en/Pages/copyright.aspx
"""
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'python-wrapper'))
from crc_covlib.helper import itur_data


if __name__ == '__main__':
    install_core = False
    install_default_helper = False
    install_montly_p840 = False
    install_monthly_p2145 = False

    print("\nNOTICE: This script accesses the International Telecommunication Union (ITU)\n" \
          "website to download data files for the user's personal use.")
    
    if sys.argv[1:] and sys.argv[1].lower() == "-d": # -d or -D for default install
        install_core = True
        install_default_helper = True
    elif sys.argv[1:] and sys.argv[1].lower() == "-m": # -m or -M for minimal install
        install_core = True
    else: # prompt user
        user_input_0 = input("\nPlease choose an option for the download and installation of ITU files.\n" \
                             "The download process may take several minutes.\n" \
                             "Required disk space is shown between parenthesis.\n"
                             "[D] Default (1.02 GB)  [M] Minimal (3.5 MB)  [C] Custom  [E] Exit: ")
        
        if user_input_0.lower() == 'd' or user_input_0.lower() == "default":
            install_core = True
            install_default_helper = True
        elif user_input_0.lower() == 'm' or user_input_0.lower() == "minimal":
            install_core = True
        elif user_input_0.lower() == 'c' or user_input_0.lower() == "custom":
            user_input_1 = input("\nDownload main ITU digital maps (requires 3.5 MB of disk space)?" \
                                 "\n[Y] Yes  [N] No  (default=\"Y\"): ")
            if user_input_1.lower() == 'y' or user_input_1.lower() == "yes" or user_input_1 == "":
                install_core = True

            user_input_2 = input("\nDownload ITU data for the crc_covlib.helper python sub-package (requires 1.02 GB of disk space)?" \
                                 "\n[Y] Yes  [N] No  (default=\"Y\"): ")
            if user_input_2.lower() == 'y' or user_input_2.lower() == "yes" or user_input_2 == "":
                install_default_helper = True

            user_input_3 = input("\nInclude monthly statistics from ITU-R P.840 (requires 1.46 GB of disk space)?" \
                                 "\n[Y] Yes  [N] No  (default=\"N\"): ")
            if user_input_3.lower() == 'y' or user_input_3.lower() == "yes":
                install_montly_p840 = True

            user_input_4 = input("\nInclude monthly statistics from ITU-R P.2145 (requires 7.56 GB of disk space)?" \
                                 "\n[Y] Yes  [N] No  (default=\"N\"): ")
            if user_input_4.lower() == 'y' or user_input_4.lower() == "yes":
                install_monthly_p2145 = True
        else:
            exit()

    print('')

    if install_core == True:
        itur_data.DownloadCoreDigitalMaps(os.path.join(script_dir, 'data', 'itu-proprietary')) # download for C++
        itur_data.DownloadCoreDigitalMaps(None) # download for crc_covlib python package

    if install_default_helper == True:
        itur_data.DownloadITURP453Data()
        itur_data.DownloadITURP530Data()
        itur_data.DownloadITURP676Data()
        itur_data.DownloadITURP837Data()
        itur_data.DownloadITURP839Data()
        itur_data.DownloadITURP840AnnualData()
        itur_data.DownloadITURP1511Data()
        itur_data.DownloadITURP2001Data()
        itur_data.DownloadITURP2145AnnualData()

    if install_montly_p840 == True:
        itur_data.DownloadITURP840MonthtlyData()

    if install_monthly_p2145 == True:
        itur_data.DownloadITURP2145MonthtlyData()


    print('\ndownload/install completed')