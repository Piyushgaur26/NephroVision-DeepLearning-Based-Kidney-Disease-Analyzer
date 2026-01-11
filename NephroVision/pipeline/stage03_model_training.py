from NephroVision.config.configuration import ConfigurationManager
from NephroVision.components.model_training import Training
from NephroVision import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):

        # Get training configuration
        config = ConfigurationManager()
        training_config = config.get_training_config()

        # Initialize training component
        training = Training(config=training_config)

        # Load prepared model from disk (CRITICAL)
        training.load_model()

        # Prepare data generators
        training.train_valid_generator()

        # Train
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")

        obj = ModelTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
