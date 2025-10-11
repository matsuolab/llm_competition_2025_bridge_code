from huggingface_hub import login, snapshot_download
import yaml, os, sys


# import settings
currdir=os.getcwd()
yml_name = sys.argv[1]
yml_path=os.path.join(currdir,yml_name)
print('yaml path {pth}'.format(pth=yml_path))

with open(yml_path, encoding='utf-8')as f:
     config = yaml.safe_load(f)
print('Config file loaded')

# Login to Huggingface in case restricted model
HF_TOKEN=config['data']['hf_token']
login(HF_TOKEN)

# create directory to store models
os.makedirs(config['model']['model_stored_path'], exist_ok=True)

# Download model
#model_name = sys.argv[2]
model_name = config['model']['model_name']
snapshot_download(
    repo_id=model_name,
    revision="main",  # or a specific commit for reproducibility
    local_dir=os.path.join(config['model']['model_stored_path'],model_name.split("/")[-1]),
    local_dir_use_symlinks=False
)

print('Download completed!')
