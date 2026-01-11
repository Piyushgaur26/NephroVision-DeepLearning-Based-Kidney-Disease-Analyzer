from NephroVision.config.configuration import ConfigurationManager
from NephroVision.components.model_evaluation import Evaluation
from NephroVision import logger

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()
        evaluation.save_score()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")

        obj = EvaluationPipeline()
        obj.main()

        logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
