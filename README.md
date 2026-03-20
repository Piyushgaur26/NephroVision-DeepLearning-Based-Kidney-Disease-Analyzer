# Kidney Disease Classification - End-to-End MLOps & Deep Learning

This repository contains an end-to-end deep learning project for classifying kidney CT scan images as either normal or containing a tumor. This project implements robust MLOps practices, integrating modular coding, pipeline tracking with DVC, experiment tracking with MLflow, and automated CI/CD deployment to AWS using GitHub Actions and Docker.

## 📌 Project Overview
The objective of this project is to build a robust image classification system using a custom VGG16 base model architecture to detect kidney tumors from CT scans. The project emphasizes production-grade architecture, featuring proper logging, exception handling, and CI/CD pipelines. 

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python 3.8
* **Deep Learning Framework:** TensorFlow 2.x (Keras)
* **Web Framework:** Flask
* **MLOps & Pipeline Tracking:** DVC (Data Version Control)
* **Experiment Tracking & Model Registry:** MLflow connected via DagsHub
* **CI/CD & Deployment:** GitHub Actions, Docker, AWS (EC2 & ECR)

## 🏗️ Project Workflows
The development of this project strictly adhered to the following step-by-step workflow for each component:
1. Update `config.yaml`
2. Update `secrets.yaml` (Optional)
3. Update `params.yaml`
4. Update the entity
5. Update the configuration manager in `src/config`
6. Update the components (Data Ingestion, Prepare Base Model, Model Training, Model Evaluation)
7. Update the pipeline
8. Update `main.py`
9. Update `dvc.yaml`
10. Update `app.py`

## 🚀 How to Run the Project Locally

**1. Clone the repository**
```bash
git clone <your-github-repository-link>
cd <your-repository-folder>
```

**2. Create a conda environment and activate it**
```bash
conda create -n kidney python=3.8 -y
conda activate kidney
```

**3. Install requirements**
```bash
pip install -r requirements.txt
```

**4. DagsHub & MLflow Setup (Required for Experiment Tracking)**
You need to set up a DagsHub repository and export your tracking credentials into your environment to log experiments. 
```bash
export MLFLOW_TRACKING_URI="<your-dagshub-tracking-uri>"
export MLFLOW_TRACKING_USERNAME="<your-username>"
export MLFLOW_TRACKING_PASSWORD="<your-password>"
```

**5. Execute the DVC Pipeline**
Instead of running standard python scripts, use DVC to execute the entire pipeline (Data Ingestion -> Base Model Prep -> Training -> Evaluation). DVC ensures that unchanged stages are skipped, saving computational power.
```bash
dvc repro
```
*Note: To visualize your pipeline dependency graph, you can run `dvc dag`.*

**6. Run the Flask Web App**
```bash
python app.py
```
*Navigate to `http://localhost:8080` in your browser to access the web interface and upload CT scan images for prediction.*

## ☁️ AWS CI/CD Deployment Steps
The project uses GitHub Actions to automate the build and deployment of a Docker container to an AWS EC2 instance.

1. **AWS Setup:**
   * Create an IAM User with the following policies: `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`.
   * Save the Access Key ID and Secret Access Key.
   * Create an ECR (Elastic Container Registry) repository to store the Docker image.
   * Launch an Ubuntu EC2 Instance (T2.large recommended for Deep Learning tasks).
2. **EC2 Configuration:**
   * SSH into the EC2 machine and install Docker.
   * Add the EC2 instance as a Self-Hosted Runner in your GitHub Repository settings (Actions -> Runners).
3. **GitHub Secrets:**
   Navigate to your repository settings and add the following repository secrets:
   * `AWS_ACCESS_KEY_ID`
   * `AWS_SECRET_ACCESS_KEY`
   * `AWS_REGION` (e.g., ap-south-1)
   * `AWS_ECR_LOGIN_URI`
   * `ECR_REPOSITORY_NAME`

Once configured, pushing code to the `main` branch will automatically trigger the CI/CD pipeline, building the Docker image, pushing it to ECR, and pulling/running it on your EC2 instance. Make sure to expose Port 8080 in the EC2 Security Group inbound rules to access the live app.
