from NephroVision import logger
from NephroVision.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from NephroVision.pipeline.stage03_model_training import ModelTrainingPipeline
from NephroVision.pipeline.stage04_model_evaluation import EvaluationPipeline
from NephroVision.pipeline.stage02_prepare_base_model import (
    PrepareBaseModelPipeline,
)


def run_stage(stage_name: str, pipeline_class, *args, **kwargs):
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class(*args, **kwargs)
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    run_stage("Data Ingestion stage", DataIngestionTrainingPipeline)
    run_stage("Prepare Base Model", PrepareBaseModelPipeline)
    full_model, _ = PrepareBaseModelPipeline().main()
    run_stage("Training", ModelTrainingPipeline, full_model=full_model)
    run_stage("Evaluation stage", EvaluationPipeline)
