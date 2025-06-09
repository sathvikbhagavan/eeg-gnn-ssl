The requirements.txt file lists all the Python libraries that the code depends on. On terminal, run the following to install them:

```
pip install -r requirements.txt
```

## Distance Based Graph


For Training: on terminal, run the following:
```
python3 train.py --data_path <data_path> --batch_size 128 --wandblog 0 --graph_type 'distance' --num_epochs 1 --best_ckpt_path <best_ckpt_path>
```
where `<best_ckpt_path>` is the directory where you want to save your best model checkpoint, and `<data_path>` is the directory where the eeg test and train data is present.

For Inference: on terminal, run the following:
```
python3 inference.py --data_path <data_path> --batch_size 128 --graph_type 'distance' --model_ckpt_path <model_ckpt_path>
```
where `<model_ckpt_path>` is the directory from where you want to load your pretrained model, and `<data_path>` is the directory where the eeg test and train data is present.

---

## Correlation Based Graph

For Training: on terminal, run the following:
```
python3 train.py --data_path <data_path> --batch_size 128 --wandblog 0 --graph_type 'correlation' --num_epochs 1 --best_ckpt_path <best_ckpt_path>
```
where `<best_ckpt_path>` is the directory where you want to save your best model checkpoint, and `<data_path>` is the directory where the eeg test and train data is present.

For Inference: on terminal, run the following:
```
python3 inference.py --data_path <data_path> --batch_size 128 --graph_type 'correlation' --model_ckpt_path <model_ckpt_path>
```
where `<model_ckpt_path>` is the directory from where you want to load your pretrained model, and `<data_path>` is the directory where the eeg test and train data is present.

---
## Hyperparameter Tuning Results 

| Configuration  | Max Macro-F1 on Val Set | 
| ------------- | ------------- | 
| Distance Graph, lr=1e-5, dropout=0, rnn_units = 64  | 0.5631 |
| Distance Graph, lr=1e-4, dropout=0, rnn_units = 64  | 0.5830 |
| Distance Graph, lr=1e-3, dropout=0, rnn_units = 64  | 0.7174 |
| Distance Graph, lr=1e-5, dropout=0.2, rnn_units = 64  | 0.5610 |
| Distance Graph, lr=1e-4, dropout=0.2, rnn_units = 64  | 0.5694 |
| Distance Graph, lr=1e-3, dropout=0.2, rnn_units = 64  | 0.6901 |
| Correlation Graph, lr=1e-3, dropout=0, rnn_units = 64  | 0.6401 |
| Correlation Graph, lr=1e-3, dropout=0, rnn_units = 128  |  0.6389 |
| Correlation Graph, lr=1e-4, dropout=0, rnn_units = 64  | 0.6082 |
| Correlation Graph, lr=1e-4, dropout=0, rnn_units = 128  | 0.6373 |
