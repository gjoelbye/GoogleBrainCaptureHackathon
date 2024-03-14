# How to deploy your app on GCP
$ are terminal commands

1. Access your workbench on Google Cloud and open a terminal
2. ```$ cd GoogleBrainCaptureHackathon```
Create virtual environment and download dependencies
3. ```$ python3 -m venv medtech ```
4. ```$ source medtech/bin/activate ```
5. ```$ pip install -r requirements.txt ```
Create Docker image and deploy service in Cloud run
6. ```$ GCP_PROJECT='your_project_id'``` 
7. ```$ GCP_REGION='europe-west1'```
8. ```$ AR_REPO='hackathon'```
9. ```$ SERVICE_NAME='streamlit-app'```
10. ```$ gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker```
11. ```$ gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"``` (this might take some time)
Launch your app!
12. ```$ gcloud run deploy "$SERVICE_NAME" --port=8080 --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" --allow-unauthenticated --region=$GCP_REGION --platform=managed --project=$GCP_PROJECT --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION```




