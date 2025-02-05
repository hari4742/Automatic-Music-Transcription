import hydra
from omegaconf import DictConfig
import multiprocessing
import logging
from src.processing.pipelines.maestro_pipeline import run_pipeline
from src.utils.logger import setup_logging


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the pipeline.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    setup_logging(cfg.logs.output_dir, cfg.logs.output_filename)
    logger = logging.getLogger(__name__)
    logger.info("Starting Maestro Processing Pipeline...")
    run_pipeline(cfg)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Required for Windows/macOS
    main()
