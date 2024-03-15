# How to deploy your app on GCP
$ are terminal commands

1. Access your workbench on Google Cloud and open a terminal. Create virtual environment and download dependencies
2. ```$ python3 -m venv medtech ```
3. ```$ source medtech/bin/activate ```
4. ```$ pip install -r requirements.txt ```
5. ```$ cd GoogleBrainCaptureHackathon```

Build Artifacts Registry (AR) repository

6. ```$ GCP_PROJECT='your_project_id'``` 
7. ```$ GCP_REGION='europe-west1'```
8. ```$ AR_REPO='hackathon'```
9. ```$ SERVICE_NAME='streamlit-app'```
10. ```$ gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker```

(GCP_PROJECT and GCP_REGION must match your ID and region, but you can name AR_REPO and SERVICE_NAME whatever you want.)

Alternatively you can create the AR repository manually: search for "artifacts registry" in the search bar, press "CREATE REPOSITORY":
- Name your repository
- Under Format, choose "Docker"
- Select your desired region
- Press "CREATE"

Build docker image manually and push to AR repostiory

11. ```$ docker build -t "docker-image" .``` (this might take some time)
12. After everything is finished, the console should print: "Successfully built {DOCKER_IMAGE}"
13. ```$ docker tag {DOCKER_IMAGE} {GCP_REGION}-docker.pkg.dev/{GCP_PROJECT}/{AR_REPO}/{SERVICE_NAME}:latest```

where {DOCKER_IMAGE}, {GCP_REGION}, {GCP_PROJECT}, {AR_REPO}, {SERVICE_NAME} should be the same as the ones you used to create your AR repository.

14. ```$ docker push {GCP_REGION}-docker.pkg.dev/{GCP_PROJECT}/{AR_REPO}/{SERVICE_NAME}```

Launch your app!

15. ```$ gcloud run deploy "$SERVICE_NAME" --port=8080 --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" --allow-unauthenticated --region=$GCP_REGION --platform=managed --project=$GCP_PROJECT --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION```

Note: It might occur that "Setting AIM Policy" fails. In this case, you might encounter "ERROR: Forbidden. Your client does not have permission to get URL / from this server". To fix this, do the following:

16. Go to the search bar and search "Cloud run"
17. Find your {SERVICE_NAME} in the list of Services. Under "Authentication", it should say "Require authentication".
18. Tick the box next to your {SERVICE_NAME}, then click "PERMISSIONS"   
19. A slide window should open: Click "ADD PRINCIPAL". Under "New principals", add the principal "allUsers". Under "Select a Role", add the role "Cloud run invoker".
20. Press "SAVE" and then "ALLOW PUBLIC ACCESS". 
21. Under "Authentication", it should now have changed to "Allow unauthenticated".

Finally, your app might need some more memory to run. To increase the memory of your instance, do the following:

22. ```$  gcloud run services update {SERVICE_NAME} --memory 4G```

