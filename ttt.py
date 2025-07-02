import os
import sagemaker

config_dir = os.path.join(os.path.dirname(sagemaker.__file__), "image_uri_config")
print("Config dir:", config_dir)
print("Files:", os.listdir(config_dir))
