select count(data_params), data_params, model_params, val_acc, max(val_f1macro), test_acc, test_f1macro, flops from results where model_params like '%"hidden_size": 128%'  group by data_params;