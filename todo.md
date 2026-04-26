- tests to run
    - feed back model's own predictions to itself during single-flight evaluation
    - make latlon relative to origin point
    - try training on latlon only, or maybe altitude only / AKA 1 LSTM per state variable
    - vanillify the LSTM model and see what happens

- what modifications to write
    - bidirectional LSTM?
    - RLSTM?

- visualizations
    - flip through time horizon predictions as more data is received
    - graph the 2D graph ground truth vs time horizon prediction
