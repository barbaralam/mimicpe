

download_some_ekg:
	wget -P ./data https://physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238.dat
	wget -P ./data https://physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238.hea

setup:
	pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
	