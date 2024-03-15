# we print the nohup.out file here, because the file is too large to `cat` in the terminal
with open("nohup.out", "r", encoding="utf-8") as file:
    data = file.read()
    file_size = len(data) / 1024 / 1024
    print(f"nohup.out size: {file_size:.2f} MB")
    data = data.strip().split("\n")
    print(data[-3])
    print(data[-1])

"""output
(GPT) root@asr:~/minasr# python read_nohup.py 
nohup.out size: 46.18 MB
Dumping kmeans label for train-other-500 for 985-126228-51: 100%|██████████| 148688/148688 [3:07:26<00:00, 13.22it/s]
"""