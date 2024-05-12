# MRFNN
For environment configuration, please refer to requirements.txt
# Enterprise Risk Prediction Project

This repository contains a suite of tools and models designed to predict and analyze enterprise risks by leveraging advanced graph theory and machine learning techniques.

## Project Architecture

### 01_knowgraph
**Create Enterprise Violation Knowledge Graph**
This module is responsible for constructing a knowledge graph that captures the intricate relationships and potential violations within an enterprise. It uses semantic technologies to interconnect various data points and entities, providing a comprehensive view of the enterprise's compliance landscape.

### 02_min_graph
**Create Enterprise Risk Association Network**
The `min_graph` module focuses on creating a minimal, yet effective, representation of the risk association network. It simplifies the complex web of interactions into a more manageable graph that can be used for further analysis and risk prediction.

### 03_hgnn
**Heterogeneous Graph Neural Network for Risk Feature Integration**
This component employs a Heterogeneous Graph Neural Network (HGNN) to integrate and process multi-source risk features. The HGNN is adept at handling diverse types of data and is capable of predicting potential violations by understanding the complex interactions within the enterprise.

### 04_wgan-GP
**Generate Violation Vectors for Class Distribution Balancing**
The `wgan-GP` module utilizes a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate synthetic violation vectors. This approach helps in balancing the class distribution, which is crucial for training robust machine learning models, especially when dealing with imbalanced datasets.

## Getting Started

To get started with the project, clone the repository and follow the instructions in the `SETUP.md` file.

```bash
git clone https://github.com/[username]/EnterpriseRiskPrediction.git
cd EnterpriseRiskPrediction
