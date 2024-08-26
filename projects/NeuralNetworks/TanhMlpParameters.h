#pragma once

#include "MlpParameters.h"

struct TanhMlpParameters
{
    TanhMlpParameters();

    static const size_t INPUT_SIZE = 1 ;
    static const size_t OUTPUT_SIZE = 1 ;
    static const size_t HIDDEN_SIZE = 16 ;
    static const size_t NUM_LAYERS = 3 ;

    MlpParameters<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS> params;
};
