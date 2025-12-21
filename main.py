from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline


def run_stage(stage_name: str, pipeline_class):
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    run_stage("Data Ingestion stage", DataIngestionTrainingPipeline)
    run_stage("Prepare Base Model", PrepareBaseModelTrainingPipeline)
    run_stage("Training", ModelTrainingPipeline)
    run_stage("Evaluation stage", EvaluationPipeline)
