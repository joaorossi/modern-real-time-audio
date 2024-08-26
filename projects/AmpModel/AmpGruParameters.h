#pragma once

#include "GruParameters.h"

struct AmpGruParameters
{
    AmpGruParameters();

    static const size_t INPUT_SIZE = 3 ;
    static const size_t OUTPUT_SIZE = 1 ;
    static const size_t HIDDEN_SIZE = 16 ;
    GruParameters<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE> params;
};
