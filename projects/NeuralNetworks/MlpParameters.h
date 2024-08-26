#pragma once
#include <cstddef>


template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t HIDDEN_SIZE, size_t NUM_LAYERS>
struct MlpParameters
{
    static_assert(NUM_LAYERS > 2, "Number of layers must be greater than 2");

    float weight_in[HIDDEN_SIZE * INPUT_SIZE];
    float bias_in[HIDDEN_SIZE];

    float weight_hidden[(NUM_LAYERS - 2) * HIDDEN_SIZE * HIDDEN_SIZE];
    float bias_hidden[(NUM_LAYERS - 2) * HIDDEN_SIZE];

    float weight_out[OUTPUT_SIZE * HIDDEN_SIZE];
    float bias_out[OUTPUT_SIZE];
};
