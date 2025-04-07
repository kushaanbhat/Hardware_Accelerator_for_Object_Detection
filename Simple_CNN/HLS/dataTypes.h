#ifndef DATATYPES_H
#define DATATYPES_H

#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "cnn.h"


// AXI Interface to Floating-point conversion helper function
inline float axi_to_float(AXI_VAL &input) {
    union { unsigned int i; float f; } converter;
    converter.i = input.data;
    return converter.f;
}

// Floating-point to AXI Interface conversion helper function
inline AXI_VAL float_to_axi(float val) {
    AXI_VAL output;
    union { unsigned int i; float f; } converter;
    converter.f = val;
    output.data = converter.i;
    output.last = 0;
    return output;
}

// Float to ap_fixed
inline data float_to_fixed(float val) {
    return (data)val;
}

// ap_fixed to Float
inline float fixed_to_float(data val) {
    return (float)val;
}

// ap_fixed to AXI
inline AXI_VAL fixed_to_axi(data val) {
    AXI_VAL axi;
    union { float f; uint32_t u; } converter;
    converter.f = (float)val;
    axi.data = converter.u;
    axi.keep = -1;
    axi.strb = -1;
    axi.last = 0; // Keep this if you're using streams
    return axi;
}


// AXI to ap_fixed
inline data axi_to_fixed(AXI_VAL axi) {
    union { float f; uint32_t u; } converter;
    converter.u = axi.data;
    return (data)(converter.f);
}

#endif // DATA_CONVERSION_H
