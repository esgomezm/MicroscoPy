import os, shutil

for dataset_name in os.listdir(os.path.join('results')): 
    for model_name in os.listdir(os.path.join('results', dataset_name)): 
        for scale_folder in os.listdir(os.path.join('results', dataset_name, model_name)): 
            for config in os.listdir(os.path.join('results', dataset_name, model_name, scale_folder)):
                config_path = os.path.join('results', dataset_name, model_name, scale_folder, config)
                if 'predicted_images' in os.listdir(config_path) and 'test_metrics' in os.listdir(config_path):
                    print(config_path)
                # if 'test_metrics' in os.listdir(config_path):
                #     # print(config_path)
                #     shutil.rmtree(os.path.join(config_path, 'test_metrics'))

