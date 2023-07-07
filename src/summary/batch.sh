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
python src/summary/generation_plot.py --exp_names logging/REDD_424/TSNET_pareto_0706_2104 logging/REDD_424/Bitcn_0703_1744 \
    logging/REDD_424/CNNLSTM_0703_1716 logging/REDD_424/ML_LSTMAE_0703_1717 --output final_424 --format $format

python src/summary/generation_plot.py --exp_names logging/UKDALE_424/TSNET_pareto_0706_1652 logging/UKDALE_424/Bitcn_0703_1744 logging/UKDALE_424/CNNLSTM_0703_1707 \
    logging/UKDALE_424/LSTMAE_0703_1701 --output final_424 --format $format

python src/summary/generation_plot.py --exp_names logging/REDD_424_5/TSNET_pareto_0706_2129 logging/REDD_424_5/Bitcn_0703_1935/ \
        logging/REDD_424_5/CNNLSTM_0703_2136 logging/REDD_424_5/ML_LSTMAE_0703_2136 --output final_424_5 --format $format