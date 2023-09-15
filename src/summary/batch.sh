format=pdf
# python src/summary/compare.py --exp_dir logging/REDD_424
# python src/summary/compare.py --exp_dir logging/UKDALE_424
# python src/summary/generation_plot.py --exp_dir logging/REDD --exp_names ML_GE_0620_0139
# python src/summary/compare.py --exp_dir logging/REDD_424_5
# python src/summary/get_best_config.py --exp_name logging/UKDALE_424/GDh2_0703_0311/ --output run/ukdale/multilabel/best_pf_f1macro_424.json --pf True
# python src/summary/get_best_config.py --exp_name logging/REDD_424/GD_0702_0901/ --output run/redd/multilabel/best_pf_f1macro_424_h3.json --pf True
# python src/summary/get_best_config.py --exp_name logging/REDD_424/GD_0703_0312/ --output run/redd/multilabel/best_pf_f1macro_424_h1.json --pf True

# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GD_0702_0901/ --output final_424 --format $format
# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GD_0703_0312/ --output final_424 --format $format
# python src/summary/generation_plot.py --exp_dir logging/UKDALE_424/GDh2_0703_0311/ --output final_424 --format $format
# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GDL_0709_0359 --output final_424 --format $format
# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GDL_h3_0709_0821/ --output final_424 --format $format


# python src/summary/generation_plot.py --exp_names logging/REDD_424/TSNET_pareto_0706_2104 logging/REDD_424/Bitcn_0703_1744 \
#     logging/REDD_424/CNNLSTM_0703_1716 logging/REDD_424/ML_LSTMAE_0703_1717 --output final_424 --format $format

# python src/summary/generation_plot.py --exp_names logging/REDD_424/TSNET_at_lstm_0709_1708 logging/REDD_424/Bitcn_0703_1744 \
#     logging/REDD_424_BK/CNNLSTM_0707_2253 logging/REDD_424/ML_LSTMAE_0703_1717 --output final_424_test --format $format

# python src/summary/generation_plot.py --exp_names logging/REDD_311/TSNET_0708_1907 logging/REDD_311/Bitcn_0708_1848 \
#     logging/REDD_311/CNNLSTM_0708_1848 logging/REDD_311/LSTMAE_0708_1901 --output final_311 --format $format

# python src/summary/generation_plot.py --exp_names logging/REDD_311_5/TSNET_0708_1924 logging/REDD_311_5/Bitcn_0708_1924 \
#     logging/REDD_311_5/CNNLSTM_0708_1924 logging/REDD_311_5/LSTMAE_0708_1924 --output final_311_5 --format $format

# python src/summary/generation_plot.py --exp_names logging/UKDALE_424/TSNET_pareto_0706_1652 logging/UKDALE_424/Bitcn_0703_1744 logging/UKDALE_424/CNNLSTM_0703_1707 \
#     logging/UKDALE_424/LSTMAE_0703_1701 --output final_424 --format $format

# python src/summary/generation_plot.py --exp_names logging/REDD_424_5/TSNET_at_lstm_0709_1851 logging/REDD_424_5/Bitcn_0703_1935/ \
#         logging/REDD_424_5/CNNLSTM_0703_2136 logging/REDD_424_5/ML_LSTMAE_0703_2136 --output final_424_5 --format $format


# -------- lstm ---------
# python src/summary/generation_plot.py --exp_dir logging/UKDALE_424/GDL_h2_0710_0040 --output final_lstm --format $format --task walltime
# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GDL_0710_0040 --output final_lstm --format $format --task walltime

# python src/summary/generation_plot.py --exp_dir logging/UKDALE_424/GDL_h2_0710_0040 --output final_lstm --format $format
# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GDL_0710_0040 --output final_lstm --format $format

# python src/summary/generation_plot.py --exp_dir logging/UKDALE_424/GDL_h2_0710_0040 --y_axis test_f1macro --output final_lstm --format $format
# python src/summary/generation_plot.py --exp_dir logging/REDD_424/GDL_0710_0040 --y_axis test_f1macro --output final_lstm --format $format


# python src/summary/get_best_config.py --exp_names logging/REDD_424/GDL_0710_0040/ --output run/redd/multilabel/best_f1macro_424_lstm.json 
# python src/summary/get_best_config.py --exp_names logging/REDD_424/GDL_0710_0040/ --output run/redd/multilabel/best_pf_f1macro_424_lstm.json --pf True

# python src/summary/get_best_config.py --exp_names logging/UKDALE_424/GDL_h2_0710_0040/ --output run/ukdale/multilabel/best_f1macro_424_lstm.json 
# python src/summary/get_best_config.py --exp_names logging/UKDALE_424/GDL_h2_0710_0040/ --output run/ukdale/multilabel/best_pf_f1macro_424_lstm.json --pf True

# python src/summary/generation_plot.py --exp_names logging/REDD_lstm/TSNET_pf_0710_2219 logging/REDD_424/Bitcn_0703_1744 logging/REDD_424_BK/CNNLSTM_0707_2253 \
#     logging/REDD_424/ML_LSTMAE_0703_1717 logging/REDD_lstm/ML_MLkNN_0703_1742 logging/REDD_lstm/ML_MLSVM_0703_1742 --output final_lstm --format $format --y_axis test_f1macro --task all_cnf

python src/summary/generation_plot.py --exp_names logging/REDD_lstm/TSNET_pf_0710_2219 logging/REDD_424/Bitcn_0703_1744 logging/REDD_424_BK/CNNLSTM_0707_2253 \
    logging/REDD_424/ML_LSTMAE_0703_1717 --output final_lstm --format $format --y_axis test_f1macro --task comp_pf

# python src/summary/generation_plot.py --exp_names logging/REDD_lstm/TSNET_pf_0710_2219 logging/REDD_424/Bitcn_0703_1744 logging/REDD_424_BK/CNNLSTM_0707_2253 \
#     logging/REDD_424/ML_LSTMAE_0703_1717 --output final_lstm --format $format --y_axis val_f1macro --task comp_pf

# python src/summary/generation_plot.py --exp_names logging/UKDALE_lstm/TSNET_pf_0711_0112 logging/UKDALE_424/Bitcn_0703_1744 logging/UKDALE_424/CNNLSTM_0703_1707 \
#     logging/UKDALE_424/LSTMAE_0703_1701 logging/UKDALE_lstm/MLkNN_0703_1741 logging/UKDALE_lstm/MLSVM_0703_1740 --output final_lstm --format $format --y_axis test_f1macro --task all_cnf

python src/summary/generation_plot.py --exp_names logging/UKDALE_lstm/TSNET_pf_0711_0112 logging/UKDALE_424/Bitcn_0703_1744 logging/UKDALE_424/CNNLSTM_0703_1707 \
    logging/UKDALE_424/LSTMAE_0703_1701 --output final_lstm --format $format --y_axis test_f1macro --task comp_pf

# python src/summary/compare.py --exp_dir logging/REDD_lstm
# python src/summary/compare.py --exp_dir logging/UKDALE_lstm