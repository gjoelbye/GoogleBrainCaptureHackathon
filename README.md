# Copenhagen Medtech X Google Cloud X BrainCapture Hackathon

<img src="assets/logos/cm_logo.png" width="30%"> <img src="assets/logos/gcp_logo.png" width="30%"> <img src="/assets/logos/bc_logo.png" width="30%">

## Introduction

Welcome to the BrainCapture EEG Analysis Hackathon! 

This hackathon challenges participants to leverage artificial intelligence techniques to enhance EEG analysis. BrainCapture utilizes an advanced transformer model to map segments of EEG data into a latent, high-dimensional space, encapsulating pertinent information within the EEG signals. Your task is to create a cloud pipeline for analyzing these latent representations.

## What are EEGs?
Electroencephalography (EEG) is a non-invasive method for recording electrical activity in the brain. It is commonly used to diagnose epilepsy, sleep disorders, and other brain-related conditions. EEGs are also used in research to study brain activity and cognitive processes. By analyzing EEG data, researchers can gain insights into brain function and develop new treatments for neurological disorders. An example of an EEG signal is shown below.

<img src="assets/eeg_example.jpg" width="900">

## Scope of the hackathon

The aim of this hackathon is to develop a cloud-based pipeline for analyzing EEG data using artificial intelligence techniques. Specifically, participants are tasked with the following:

1. Import EEG sessions and segment them into smaller, relevant windows.
2. Transform these windows into latent representations.
3. Analyze the latent representations using a data analysis or machine learning pipeline.
4. Present the findings via a cloud-based platform.

## Approaches

The challenge can be approached in various ways. Some suggested approaches include:

- Implementing machine learning algorithms for EEG data analysis.
- Developing visualization techniques for exploring latent representations.
- Creating a web-based platform for interactive data exploration.

## Provided Materials

To aid participants in the hackathon, the following materials will be provided:

- Two EEG datasets (BC Bhutan and TUAR) (see [data](data/))
- A model for encoding EEG data into latent representations (see [BENDR Model](models/))
- Sample code for simple learning algorithms (e.g., KNN) and EEG data visualizations (see [Simple Learning Algorithm Demo](demos/knn.ipynb) and [EEG Data Visualization Demo](demos/visualizations.ipynb))
- Sample code for deploying a simple website (see [Deployment Demo](demos/app/))
- A guide for deploying to the Google Cloud Platform (see [GCP Deployment Guide](docs/gcp_deployment.md))

## Evaluation Criteria

Submissions will be evaluated based on the following criteria:

1. Technical ability and findings
2. Innovation and creativity
3. Utility for BrainCapture
4. Efficiency of the cloud deployment pipeline
5. Quality of the pitch

## Getting Started/Further Reading

For detailed instructions on setting up the project and accessing provided materials, refer to the [Getting Started Guide](docs/getting_started.md).

For deploying your solution to the Google Cloud Platform, follow the steps outlined in the [GCP Deployment Guide](docs/gcp_deployment.md).

## Getting help and support

If you have any questions or need help with the hackathon, please reach out to the organizers or post your question in the Discord. We also encourage you to collaborate with other teams to solev issues and share ideas.

## License

This project is licensed under the [LICENSE](LICENSE) file.
