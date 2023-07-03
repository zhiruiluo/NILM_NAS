partition=backfill
python run/redd/multilabel/run_bitcn.py --num_nodes=2 --partition=$partition --log_dir='logging/REDD/'
python run/redd/multilabel/run_cnnlstm.py --num_nodes=2 --partition=$partition --log_dir='logging/REDD/'
python run/redd/multilabel/run_lstmae.py --num_nodes=2 --partition=$partition --log_dir='logging/REDD/'
python run/redd/multilabel/run_mlknn.py --num_nodes=2 --partition=$partition --log_dir='logging/REDD/'
python run/redd/multilabel/run_mlsvm.py --num_nodes=2 --partition=$partition --log_dir='logging/REDD/'
python run/redd/multilabel/run_genetic_tsnet_ray.py --num_nodes=4 --partition=epscor --log_dir='logging/REDD/'