#pragma once
#include <cstddef>


template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t HIDDEN_SIZE>
struct RnnParameters
{
    float weight_ih[HIDDEN_SIZE * INPUT_SIZE];
    float bias_ih[HIDDEN_SIZE];

    float weight_hh[HIDDEN_SIZE * HIDDEN_SIZE];
    float bias_hh[HIDDEN_SIZE];

    float weight_output[OUTPUT_SIZE * HIDDEN_SIZE];
    float bias_output[OUTPUT_SIZE];
};
