import hydra
from hle_benchmark import run_judge_results

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    run_judge_results.main(cfg)

if __name__ == "__main__":
    main()