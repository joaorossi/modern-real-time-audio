#pragma once

#include <cstddef>
#include <cmath>

#include "RnnParameters.h"


template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t HIDDEN_SIZE>
class Rnn
{
public:
    Rnn()
    {
        memset(weight_ih, 0, sizeof(weight_ih));
        memset(bias_ih, 0, sizeof(bias_ih));

        memset(weight_hh, 0, sizeof(weight_hh));
        memset(bias_hh, 0, sizeof(bias_hh));

        memset(weight_output, 0, sizeof(weight_output));
        memset(bias_output, 0, sizeof(bias_output));

        reset_state();
    }

    void process(float * const * output, const float * const * input, size_t num_samples)
    {
        float new_state[HIDDEN_SIZE];

        for (size_t n = 0; n < num_samples; ++n)
        {
            // add contribution from input
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
                // initialise with bias
                new_state[i] = bias_ih[i];
                // matrix-vector multiply
                for (size_t j = 0; j < INPUT_SIZE; ++j)
                {
                    new_state[i] += weight_ih[i][j] * input[n][j];
                }
            }
            // add contribution from state
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
                // add bias
                new_state[i] += bias_hh[i];
                // matrix-vector multiply
                for (size_t j = 0; j < HIDDEN_SIZE; ++j)
                {
                    new_state[i] += weight_hh[i][j] * state[j];
                }
            }
            // apply nonlinearity and update state for next sample
            for (size_t i = 0; i < HIDDEN_SIZE; ++i)
            {
                state[i] = std::tanh(new_state[i]);
            }
            // compute output
            for (size_t i = 0; i < OUTPUT_SIZE; ++i)
            {
                output[n][i] = bias_output[i];
                for (size_t j = 0; j < HIDDEN_SIZE; ++j)
                {
                    output[n][i] += weight_output[i][j] * state[j];
                }
            }
        }
    }

    void load_parameters(RnnParameters<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE> params)
    {
        memcpy(weight_ih, params.weight_ih, sizeof(weight_ih));
        memcpy(bias_ih, params.bias_ih, sizeof(bias_ih));

        memcpy(weight_hh, params.weight_hh, sizeof(weight_hh));
        memcpy(bias_hh, params.bias_hh, sizeof(bias_hh));

        memcpy(weight_output, params.weight_output, sizeof(weight_output));
        memcpy(bias_output, params.bias_output, sizeof(bias_output));
    }

    void reset_state()
    {
        memset(state, 0, sizeof(state));
    }

private:
    // model parameters
    float weight_ih[HIDDEN_SIZE][INPUT_SIZE];
    float bias_ih[HIDDEN_SIZE];

    float weight_hh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bias_hh[HIDDEN_SIZE];

    float weight_output[OUTPUT_SIZE][HIDDEN_SIZE];
    float bias_output[OUTPUT_SIZE];

    // rnn state
    float state[HIDDEN_SIZE];
};
