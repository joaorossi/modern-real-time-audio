#pragma once

#include <cstddef>
#include <cmath>

#include "GruParameters.h"


template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t HIDDEN_SIZE>
class Gru
{
public:
    Gru()
    {
        memset(weight_ih_r, 0, sizeof(weight_ih_r));
        memset(weight_ih_z, 0, sizeof(weight_ih_z));
        memset(weight_ih_n, 0, sizeof(weight_ih_n));

        memset(bias_ih_r, 0, sizeof(bias_ih_r));
        memset(bias_ih_z, 0, sizeof(bias_ih_z));
        memset(bias_ih_n, 0, sizeof(bias_ih_n));

        memset(weight_hh_r, 0, sizeof(weight_hh_r));
        memset(weight_hh_z, 0, sizeof(weight_hh_z));
        memset(weight_hh_n, 0, sizeof(weight_hh_n));

        memset(bias_hh_r, 0, sizeof(bias_hh_r));
        memset(bias_hh_z, 0, sizeof(bias_hh_z));
        memset(bias_hh_n, 0, sizeof(bias_hh_n));

        memset(weight_output, 0, sizeof(weight_output));
        memset(bias_output, 0, sizeof(bias_output));

        reset_state();
    }

    float sigmoid(float x) const
    {
        // TODO 1) Implement the sigmoid function
    }

    void process(float * const * output, const float * const * input, size_t num_samples)
    {
        float r_gate[HIDDEN_SIZE];
        float z_gate[HIDDEN_SIZE];
        float n_gate[HIDDEN_SIZE];
        float n_hidden[HIDDEN_SIZE];

        for (size_t n = 0; n < num_samples; ++n)
        {
            // Compute r gate - this is shown here as an example
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
                // Initialise with bias_ih
                r_gate[i] = bias_ih_r[i];
                // Implement matrix-vector multiply with weight_ih and the input signal
                for (size_t j = 0; j < INPUT_SIZE; ++j)
                {
                    r_gate[i] += weight_ih_r[i][j] * input[n][j];
                }
                // Add bias_hh
                r_gate[i] += bias_hh_r[i];
                // Implement matrix-vector multiply with weight_hh and the state signal
                for (size_t j = 0; j < HIDDEN_SIZE; ++j)
                {
                    r_gate[i] += weight_hh_r[i][j] * state[j];
                }
                // Apply sigmoid activation function
                r_gate[i] = sigmoid(r_gate[i]);
            }

            // TODO 2) Implement the z gate and store the result to z_gate.
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
            }

            // TODO 3) Compute n gate - note that this is different than the r and z gates
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
                // TODO 3.1) Compute W_in * x + b_in and store result to n_gate
                // TODO 3.2) Compute W_h * h + b_h and store result to n_hidden
                // TODO 3.3) Multiply n_hidden by the r gate
                // TODO 3.4) Add n_hidden to n_gate
                // TODO 3.5) Apply tanh activation function to n_gate
            }

            // TODO 4) Compute the new state h_t = (1 - z) * n + z * h_{t-1}
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
            }

            // TODO 5) Implement the affine output layer - store final output to output[n][i]
            for (size_t i = 0; i < OUTPUT_SIZE; ++i)
            {
            }
        }
    }

    void load_parameters(GruParameters<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE> params)
    {
        memcpy(weight_ih_r, params.weight_ih_r, sizeof(weight_ih_r));
        memcpy(weight_ih_z, params.weight_ih_z, sizeof(weight_ih_z));
        memcpy(weight_ih_n, params.weight_ih_n, sizeof(weight_ih_n));

        memcpy(bias_ih_r, params.bias_ih_r, sizeof(bias_ih_r));
        memcpy(bias_ih_z, params.bias_ih_z, sizeof(bias_ih_z));
        memcpy(bias_ih_n, params.bias_ih_n, sizeof(bias_ih_n));

        memcpy(weight_hh_r, params.weight_hh_r, sizeof(weight_hh_r));
        memcpy(weight_hh_z, params.weight_hh_z, sizeof(weight_hh_z));
        memcpy(weight_hh_n, params.weight_hh_n, sizeof(weight_hh_n));

        memcpy(bias_hh_r, params.bias_hh_r, sizeof(bias_hh_r));
        memcpy(bias_hh_z, params.bias_hh_z, sizeof(bias_hh_z));
        memcpy(bias_hh_n, params.bias_hh_n, sizeof(bias_hh_n));

        memcpy(weight_output, params.weight_output, sizeof(weight_output));
        memcpy(bias_output, params.bias_output, sizeof(bias_output));
    }

    void reset_state()
    {
        memset(state, 0, sizeof(state));
    }

private:
    // model parameters
    float weight_ih_r[HIDDEN_SIZE][INPUT_SIZE];
    float weight_ih_z[HIDDEN_SIZE][INPUT_SIZE];
    float weight_ih_n[HIDDEN_SIZE][INPUT_SIZE];

    float bias_ih_r[HIDDEN_SIZE];
    float bias_ih_z[HIDDEN_SIZE];
    float bias_ih_n[HIDDEN_SIZE];

    float weight_hh_r[HIDDEN_SIZE][HIDDEN_SIZE];
    float weight_hh_z[HIDDEN_SIZE][HIDDEN_SIZE];
    float weight_hh_n[HIDDEN_SIZE][HIDDEN_SIZE];

    float bias_hh_r[HIDDEN_SIZE];
    float bias_hh_z[HIDDEN_SIZE];
    float bias_hh_n[HIDDEN_SIZE];

    float weight_output[OUTPUT_SIZE][HIDDEN_SIZE];
    float bias_output[OUTPUT_SIZE];

    // gru state
    float state[HIDDEN_SIZE];
};
