import hydra
# vLLM用の推論モジュールを直接インポート
from benchmark import vllm_predictions

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    # vLLMの推論を直接実行
    vllm_predictions.main(cfg)

if __name__ == "__main__":
    main()