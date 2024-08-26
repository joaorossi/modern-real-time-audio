#pragma once

#include "RnnParameters.h"

struct TanhRnnParameters
{
    TanhRnnParameters();

    static const size_t INPUT_SIZE = 1 ;
    static const size_t OUTPUT_SIZE = 1 ;
    static const size_t HIDDEN_SIZE = 16 ;
    RnnParameters<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE> params;
};
