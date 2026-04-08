# Machine Learning Final Project Proposal: Trajectory Prediction with Recurrent Neural Networks
> Ivy Mahncke & Lily Wei

## Project Overview

### Project Goals



### Motivation



## Deliverables

As a final deliverable for this project, we plan to write a final report on our learning that includes the following components:

### (Context) Necessity of Trajectory Prediction for Aircraft



### (Theory) Summary of Sequential Prediction Models



### (Implementation) Codebase of Model Implementation



### (Implementation) Performance Analysis



### (Context) Implications of Model Performance and Future Work



## Timeline

We will commit to the following timeline for this project, given that we have 4 weeks to complete it.

### Week 1:

- Model(s) Selection
    - Decide what to investigate and evaluate
    - Current list to pare down:
        - RNNs/LSTMs
        - Transformers
        - CRNNs
        - GRUs
- Gather Datasets
    - paring them down to specific scenarios (how generalizable?)
    - standardizing them with same columns and features
    - consider aircraft types (fixed-wing, drone, helicopter)
- Proof-Of-Concept Trial
    - Try Lily's existing LSTM with minimal modifications
    - Set up original LSTM if useful
    - Inform pivot if necessary
- Document Initial Findings
    - Write trajectory prediction context in report
    - Write summary of underlying model theory in report

### Week 2:

- Baseline Establishment
    - Write a script to implement physical model prediction
        - Kinematics equations
        - Kalman Filtering
    - Plotting code for trajectories
- Paper Replication
    - Select papers most relevant to our methods and replicate them
    - Majority of codebase development happens
    - Initial trial, compare with physical model
- Investigate Tuning
    - Outline tunable hyperparameters and perform parameter sweeps
    - Research modifications to baseline LSTM
    - Consider implementing a hybrid model

### Week 3:

- Buffer Week
    - Going well: implement hybrid model, modify baseline LSTM
    - Going okay: continue fine-tuning parameters and improving model
    - Going bad: extra work time, debugging, referencing more papers

### Week 4:

- Code Lock
- Evaluate Model Performance
    - Compare accuracy of trajectory over time to baseline methods and ground truth
    - Investigate and perform other evaluation methods (i.e. prediction time)
- Visualizations
    - Loss decay with different model configurations
    - Trajectory/state plots showing model outputs
    - More methods of evaluations
- Deliverables Materials
    - Clean up and document codebase
    - Document methodology, performance analysis, future work
    - Finalize report

## External Resources

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

Attention-LSTM: [Attention-LSTM based prediction model for aircraft 4-D trajectory](https://www.nature.com/articles/s41598-022-19794-1) \
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

RNN vs LSTM vs GRU vs Transformers: https://www.geeksforgeeks.org/deep-learning/rnn-vs-lstm-vs-gru-vs-transformers/
