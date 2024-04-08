logfile=logs/lm_train_lightning.log
# nohup python lm_train_lightning.py > $logfile &
python utils/read_nohup.py --file_path $logfile --head 100 --tail 200