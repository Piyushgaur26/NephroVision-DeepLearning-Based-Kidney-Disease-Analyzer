from NephroVision.config.configuration import ConfigurationManager
from NephroVision.components.prepare_base_model import PrepareBaseModel
from NephroVision import logger

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):

        # Get configuration
        config = ConfigurationManager()
        prepare_config = config.get_prepare_base_model_config()

        # Prepare base model
        prepare_base_model = PrepareBaseModel(config=prepare_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


#        return prepare_base_model.full_model, prepare_config


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")

        obj = PrepareBaseModelPipeline()
        obj.main()

        logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
