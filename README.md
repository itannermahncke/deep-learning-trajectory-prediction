# Machine Learning Final Project Proposal: Trajectory Prediction with Recurrent Neural Networks
> Ivy Mahncke & Lily Wei

## Project Overview

In this project, we plan to utilize deep learning to predict future trajectories of aircraft in real-time using their incoming ADS-B flight data. We will investigate Recurrent Neural Networks (RNNs), a type of neural network used for sequential prediction and temporal data, to solve the prediction problem. After training and fine-tuning our model, we will evaluate its performance on masked flight data to the ground truth data, as well as baseline physics and modeling-based methods such as kinematics and Kalman Filtering. Our evaluation standards will be based on contextual information regarding aviation safety. Finally, we will discuss the significance of our model's relative performance in its real-world context and outline possibilities for future development.

## Repository Structure

This repository contains the following files:

```
├── src/
│   ├── lstm_model_class.py
│   ├── lstm_pipeline_module.py
│   ├── lstm_model_sweep.py
│   ├── early_stopper.py
│   ├── anomaly_detectors.kt
├── docs/
│   ├── images/
│   ├── proposal.md
│   ├── report.md
├── .gitignore
├── README.md
```

`src/` houses the source code for this project. Here's a quick descriptor of the contents:

    `lstm_model_class.py`: implementation of an LSTM neural network
    `lstm_pipeline_module.py`: script to train an LSTM on a given dataset and hyperparameters
    `lstm_model_sweep.py`: script to run a parameter sweep of several training sessions for a given LSTM
    `early_stopper.py`: helper class for halting training upon loss convergence
    `anomaly_detectors.py`:
    `utils.py`: helper functions for data preprocessing and LSTM pipeline work
    `viz.py`: helper functions for data and results visualization

`docs/` houses all documentation of this project. The two important files in here are:

    `proposal.md`: our project proposal! It contains learning goals, planned deliverables, and a project timeline.
    `report.md`: our final report! It contains background research, methodology, and results, as well as ideas for future work.


## Live External Resources

### LILY AND IVY READ
Bi-Directional LSTM: https://www.nature.com/articles/s41598-023-46914-2
LSTM on OpenSky Medium Article: https://medium.com/@albertomoccardi/airplane-traffic-prediction-atp-d180c1098027
Transformer Only: [ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace](https://arxiv.org/abs/2603.16550) \
Attention-LSTM: [Attention-LSTM based prediction model for aircraft 4-D trajectory](https://www.nature.com/articles/s41598-022-19794-1) \

### Existing Research for Replication

#### RNN/LSTM Models

Bi-LSTM: [Aircraft trajectory prediction and aviation safety in ADS-B failure conditions based on neural network](https://www.nature.com/articles/s41598-023-46914-2) \
LSTM: [Machine-Learning-Aided Trajectory Prediction and Conflict Detection for Internet of Aerial Vehicles](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9359666) \
RLSTM: [Recurrent LSTM-based UAV Trajectory Prediction with ADS-B Information](https://www.researchgate.net/publication/363210293_Recurrent_LSTM-based_UAV_Trajectory_Prediction_with_ADS-B_Information) \

#### CRNN Hybrid Models

CRNN-LSTM: [A Generalized Approach to Aircraft Trajectory Prediction via Supervised Deep Learning](https://ntrs.nasa.gov/api/citations/20220002176/downloads/Generalizing_Trajectory_Prediction_FORMATTED.pdf) \
CRNN-LSTM: [Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach](https://arxiv.org/abs/1812.11670) \
CRNN-GRU: [4D flight trajectory prediction using a hybrid Deep Learning prediction method based on ADS-B technology](https://arxiv.org/abs/2110.07774) \

#### Transformer-Based Models

[Noise robust aircraft trajectory prediction via autoregressive transformers with hybrid positional encoding](https://www.nature.com/articles/s41598-025-96512-7) \
Transformer Encoder-LSTM: [A Predictive Aircraft Trajectory Prediction Method Based on Transformer Encoder and LSTM](https://htfhyyg.spacejournal.cn/en/article/doi/10.3969/j.issn.1009-8518.2024.02.016) \
CRNN-Attention: [A deep learning framework for predicting aircraft trajectories from sparse satellite observations](https://www.nature.com/articles/s41598-025-27064-z) \
Transformer Only: [ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace](https://arxiv.org/abs/2603.16550) \

#### Other

[An Aircraft Trajectory Prediction Method Based on Trajectory Clustering and a Spatiotemporal Feature Network](https://www.mdpi.com/2079-9292/11/21/3453) \
[SkyTraceX: A Real-Time Short-Horizon AircraftTrajectory Prediction System Using GradientBoosted Telemetry Models](https://engrxiv.org/preprint/view/6745/11038) \

### Tangential Resources

Baselines: [ADS-B Trajectory Filtering Techniques](https://www.scribd.com/document/866352038/2024-Olive-etal-Filtering-techniques-ADS-B-trajectory-preprocessing) \
Baselines/Preprocessing: [Python Aircraft Data Preprocessing Library](https://github.com/xoolive/traffic) \
Data: [OpenSky: A large-scale ADS-B sensor network for research](https://ieeexplore.ieee.org/document/6846743) \
Data: [ADS-B Exchange: Unfiltered Flight Data](https://www.adsbexchange.com/) \

### Implementation

RNN vs LSTM vs GRU vs Transformers: [GeeksForGeeks](https://www.geeksforgeeks.org/deep-learning/rnn-vs-lstm-vs-gru-vs-transformers/) \
