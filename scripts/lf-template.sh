LF_PATH = <your llamafactory path>
echo "export WANDB_API_KEY=<your wandb api key>" >> ~/.bashrc
source ~/.bashrc
pip install -e ".[torch,metrics]"
pip install transformers==4.41.2 deepspeed wandb
llamafactory-cli train <path to your config>