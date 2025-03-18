#include "cnn.h"
#include <iostream>
void cnn(axi_stream &input_stream, axi_stream &output_stream);
int main() {
    // Declare input and output streams
	axi_stream input_stream;
	axi_stream output_stream;

    // Example input (random or test data)
    float test_input[30][30][3] =  {
    	    {{22, 23, 22}, {24, 25, 24}, {22, 23, 23}, {20, 20, 20}, {20, 20, 20}, {23, 23, 22}, {30, 31, 35}, {29, 31, 42}, {21, 23, 43}, {28, 28, 48}, {46, 44, 56}, {69, 65, 69}, {68, 68, 66}, {68, 69, 67}, {69, 69, 68}, {68, 68, 68}, {63, 67, 66}, {62, 66, 62}, {62, 64, 63}, {60, 63, 65}, {48, 53, 65}, {24, 30, 50}, {18, 21, 46}, {31, 30, 51}, {26, 28, 43}, {17, 20, 27}, {19, 21, 22}, {23, 24, 25}, {25, 26, 27}, {23, 25, 26}},
    	    {{22, 24, 23}, {22, 23, 24}, {21, 21, 22}, {20, 20, 20}, {24, 23, 23}, {25, 26, 26}, {26, 29, 29}, {31, 34, 36}, {25, 30, 44}, {20, 25, 44}, {30, 31, 49}, {50, 44, 56}, {64, 62, 68}, {68, 68, 71}, {69, 68, 69}, {69, 69, 69}, {68, 69, 69}, {65, 67, 68}, {60, 62, 68}, {44, 51, 62}, {24, 31, 48}, {19, 22, 42}, {23, 24, 43}, {31, 32, 44}, {24, 25, 30}, {18, 21, 23}, {18, 20, 21}, {21, 23, 24}, {21, 24, 24}, {20, 23, 22}},
    	    {{22, 23, 22}, {21, 22, 22}, {20, 21, 21}, {21, 22, 21}, {27, 26, 25}, {31, 30, 30}, {25, 26, 24}, {31, 31, 28}, {35, 36, 42}, {24, 31, 44}, {20, 24, 43}, {25, 24, 41}, {34, 32, 46}, {48, 46, 57}, {56, 52, 61}, {60, 57, 64}, {55, 55, 63}, {44, 47, 57}, {34, 40, 54}, {23, 27, 46}, {20, 22, 41}, {26, 23, 41}, {31, 31, 44}, {25, 30, 36}, {21, 24, 23}, {20, 22, 22}, {20, 22, 23}, {20, 23, 23}, {21, 23, 24}, {21, 24, 23}},
    	    {{22, 23, 22}, {20, 22, 21}, {22, 23, 22}, {25, 25, 24}, {27, 27, 25}, {30, 30, 27}, {26, 27, 25}, {28, 28, 26}, {32, 32, 29}, {33, 37, 40}, {27, 31, 47}, {21, 21, 40}, {23, 23, 43}, {22, 21, 44}, {25, 24, 48}, {27, 26, 49}, {25, 25, 48}, {21, 23, 45}, {18, 21, 43}, {20, 21, 43}, {24, 24, 44}, {31, 30, 46}, {30, 30, 37}, {19, 22, 24}, {20, 22, 22}, {23, 24, 24}, {23, 25, 25}, {23, 25, 25}, {22, 24, 24}, {22, 25, 23}},
    	    {{27, 28, 25}, {22, 24, 23}, {24, 26, 25}, {28, 29, 28}, {29, 28, 27}, {27, 28, 27}, {22, 25, 24}, {26, 27, 25}, {27, 27, 25}, {28, 31, 31}, {35, 38, 44}, {32, 33, 42}, {25, 27, 38}, {21, 24, 42}, {20, 22, 44}, {20, 22, 42}, {20, 21, 42}, {20, 20, 45}, {26, 26, 45}, {32, 30, 45}, {32, 30, 42}, {24, 27, 36}, {18, 21, 25}, {22, 20, 21}, {23, 23, 23}, {20, 23, 23}, {20, 23, 23}, {22, 25, 23}, {23, 25, 24}, {24, 26, 25}},
    	    {{26, 27, 25}, {24, 26, 25}, {24, 27, 25}, {26, 28, 27}, {28, 29, 28}, {25, 26, 26}, {21, 24, 24}, {24, 25, 24}, {25, 26, 26}, {23, 25, 25}, {24, 26, 25}, {27, 28, 29}, {33, 34, 38}, {35, 36, 50}, {31, 32, 51}, {34, 37, 53}, {34, 36, 55}, {21, 27, 53}, {28, 38, 55}, {23, 26, 36}, {22, 21, 26}, {19, 20, 22}, {19, 19, 21}, {22, 20, 21}, {21, 21, 21}, {19, 21, 21}, {21, 23, 22}, {26, 28, 24}, {26, 28, 25}, {23, 26, 26}},
    	    {{25, 25, 25}, {22, 23, 23}, {21, 24, 24}, {21, 23, 23}, {22, 23, 23}, {21, 23, 22}, {22, 24, 24}, {21, 22, 23}, {22, 23, 22}, {25, 25, 23}, {29, 29, 26}, {32, 34, 39}, {27, 28, 45}, {33, 33, 51}, {31, 32, 51}, {31, 33, 53}, {30, 32, 55}, {21, 29, 55}, {25, 39, 60}, {18, 22, 38}, {24, 23, 30}, {21, 22, 24}, {21, 21, 23}, {22, 20, 21}, {20, 20, 20}, {19, 20, 20}, {20, 21, 21}, {24, 25, 24}, {25, 27, 26}, {22, 25, 26}},
    	    {{21, 22, 22}, {23, 22, 22}, {22, 23, 24}, {21, 22, 22}, {21, 21, 21}, {21, 21, 21}, {22, 22, 22}, {21, 22, 22}, {26, 26, 27}, {36, 30, 33}, {41, 37, 42}, {29, 35, 48}, {22, 25, 49}, {25, 25, 47}, {23, 24, 45}, {23, 25, 47}, {23, 25, 50}, {18, 24, 51}, {24, 33, 58}, {28, 28, 48}, {35, 32, 44}, {28, 31, 38}, {21, 25, 29}, {20, 22, 23}, {21, 21, 21}, {21, 20, 20}, {20, 20, 20}, {22, 23, 22}, {22, 24, 23}, {21, 23, 24}},
    	    {{22, 23, 23}, {23, 23, 25}, {21, 21, 23}, {20, 20, 20}, {21, 21, 21}, {21, 21, 22}, {21, 22, 21}, {28, 26, 25}, {34, 31, 37}, {36, 34, 47}, {30, 31, 49}, {19, 24, 46}, {24, 24, 51}, {28, 25, 51}, {28, 28, 52}, {28, 28, 53}, {26, 26, 53}, {20, 24, 53}, {15, 23, 50}, {21, 23, 47}, {29, 26, 47}, {36, 36, 51}, {30, 33, 41}, {19, 24, 26}, {20, 23, 22}, {19, 21, 18}, {21, 23, 21}, {23, 25, 24}, {25, 27, 28}, {21, 24, 27}},
    	    {{21, 21, 21}, {20, 21, 21}, {19, 19, 20}, {18, 18, 18}, {20, 20, 20}, {22, 22, 22}, {25, 27, 27}, {36, 32, 33}, {39, 34, 46}, {28, 30, 47}, {24, 27, 42}, {31, 33, 48}, {44, 42, 55}, {62, 56, 67}, {67, 65, 75}, {68, 67, 77}, {61, 59, 70}, {52, 54, 66}, {34, 42, 59}, {24, 28, 49}, {23, 23, 44}, {29, 26, 46}, {34, 34, 48}, {24, 30, 39}, {20, 25, 26}, {20, 25, 24}, {21, 25, 23}, {21, 25, 24}, {22, 26, 25}, {22, 27, 28}},
    	    {{21, 21, 21}, {21, 20, 20}, {20, 21, 21}, {20, 20, 20}, {21, 20, 20}, {27, 25, 23}, {38, 33, 34}, {38, 38, 44}, {29, 31, 48}, {28, 26, 46}, {43, 39, 53}, {60, 54, 65}, {69, 66, 73}, {72, 73, 76}, {72, 74, 75}, {72, 73, 74}, {71, 71, 72}, {69, 70, 71}, {65, 68, 75}, {48, 53, 67}, {28, 33, 52}, {23, 25, 46}, {26, 26, 47}, {34, 34, 48}, {27, 28, 35}, {20, 23, 25}, {21, 23, 22}, {22, 23, 22}, {24, 25, 24}, {26, 29, 28}},
    	    {{20, 20, 21}, {20, 20, 20}, {22, 22, 21}, {22, 22, 22}, {23, 22, 20}, {29, 27, 22}, {45, 43, 48}, {34, 36, 51}, {22, 25, 47}, {29, 28, 51}, {41, 38, 57}, {52, 46, 61}, {54, 54, 65}, {49, 52, 61}, {58, 58, 64}, {68, 67, 68}, {72, 72, 70}, {71, 72, 70}, {71, 71, 71}, {65, 69, 73}, {49, 55, 68}, {27, 32, 52}, {22, 24, 48}, {34, 31, 52}, {31, 31, 48}, {20, 23, 29}, {20, 22, 22}, {21, 22, 21}, {25, 24, 24}, {27, 28, 28}},
    	    {{21, 21, 21}, {21, 21, 21}, {21, 21, 21}, {21, 21, 21}, {22, 22, 19}, {28, 31, 23}, {37, 43, 52}, {24, 27, 50}, {24, 27, 49}, {28, 34, 55}, {22, 28, 49}, {23, 24, 44}, {22, 23, 43}, {23, 24, 42}, {41, 35, 48}, {68, 60, 63}, {73, 70, 69}, {72, 72, 72}, {71, 72, 72}, {71, 71, 73}, {65, 68, 75}, {39, 48, 63}, {21, 26, 50}, {29, 26, 47}, {35, 36, 52}, {20, 25, 32}, {20, 22, 23}, {22, 22, 22}, {24, 23, 23}, {24, 25, 25}},
    	    {{20, 20, 19}, {21, 21, 22}, {21, 21, 21}, {20, 20, 20}, {23, 23, 19}, {34, 37, 27}, {34, 39, 51}, {25, 25, 54}, {32, 31, 52}, {34, 42, 61}, {24, 31, 55}, {20, 21, 48}, {19, 20, 49}, {22, 21, 48}, {40, 30, 51}, {64, 56, 61}, {69, 69, 69}, {65, 67, 70}, {63, 66, 68}, {63, 64, 65}, {68, 68, 70}, {51, 55, 64}, {29, 32, 54}, {23, 22, 42}, {31, 34, 47}, {21, 28, 35}, {20, 22, 26}, {21, 21, 23}, {21, 21, 21}, {21, 21, 21}},
    	    {{20, 20, 20}, {20, 20, 20}, {20, 20, 21}, {20, 20, 20}, {24, 24, 20}, {38, 40, 30}, {28, 33, 43}, {28, 25, 50}, {43, 38, 52}, {42, 50, 63}, {23, 31, 54}, {19, 20, 48}, {19, 20, 50}, {22, 21, 49}, {40, 30, 52}, {60, 52, 62}, {55, 59, 66}, {45, 50, 60}, {54, 58, 63}, {55, 56, 56}, {61, 61, 60}, {55, 57, 64}, {33, 36, 57}, {22, 22, 44}, {27, 32, 47}, {24, 31, 40}, {19, 23, 27}, {19, 21, 22}, {19, 21, 21}, {20, 20, 20}},
    	    {{19, 19, 19}, {19, 19, 20}, {20, 19, 20}, {21, 20, 20}, {24, 21, 18}, {36, 39, 30}, {27, 35, 44}, {29, 25, 49}, {47, 40, 51}, {45, 49, 59}, {25, 30, 50}, {20, 21, 45}, {18, 19, 45}, {20, 20, 46}, {34, 30, 51}, {46, 44, 56}, {34, 41, 50}, {24, 28, 39}, {35, 35, 40}, {35, 37, 36}, {43, 47, 44}, {51, 58, 62}, {34, 39, 61}, {22, 23, 45}, {26, 31, 47}, {25, 33, 43}, {19, 23, 27}, {20, 21, 22}, {19, 20, 20}, {19, 19, 19}},
    	    {{20, 20, 18}, {20, 19, 19}, {19, 19, 20}, {21, 20, 19}, {23, 21, 17}, {32, 35, 26}, {27, 36, 45}, {27, 25, 48}, {43, 37, 51}, {54, 54, 66}, {43, 43, 59}, {31, 28, 44}, {36, 34, 51}, {29, 28, 47}, {43, 37, 55}, {53, 48, 58}, {30, 31, 37}, {24, 23, 26}, {27, 22, 24}, {28, 23, 24}, {35, 33, 29}, {47, 52, 55}, {33, 37, 59}, {22, 23, 44}, {27, 32, 48}, {25, 33, 43}, {19, 23, 28}, {18, 20, 22}, {17, 18, 19}, {18, 19, 20}},
    	    {{22, 21, 19}, {22, 20, 21}, {21, 20, 21}, {21, 21, 20}, {21, 21, 19}, {26, 30, 24}, {31, 38, 48}, {24, 25, 47}, {37, 34, 49}, {56, 53, 63}, {54, 49, 56}, {60, 51, 57}, {61, 56, 63}, {53, 48, 57}, {56, 46, 55}, {67, 55, 61}, {50, 44, 45}, {60, 52, 46}, {56, 44, 44}, {60, 46, 49}, {56, 45, 47}, {48, 49, 57}, {28, 31, 54}, {23, 22, 44}, {30, 34, 49}, {22, 30, 39}, {16, 20, 25}, {16, 19, 20}, {17, 18, 18}, {19, 20, 20}},
    	    {{21, 21, 20}, {22, 21, 22}, {21, 21, 22}, {21, 21, 21}, {21, 21, 20}, {26, 25, 22}, {41, 39, 49}, {28, 28, 48}, {31, 30, 46}, {55, 47, 57}, {67, 60, 62}, {62, 58, 59}, {59, 58, 59}, {58, 58, 60}, {60, 57, 60}, {61, 57, 59}, {59, 56, 54}, {67, 63, 53}, {63, 56, 54}, {60, 54, 59}, {54, 50, 58}, {38, 43, 57}, {23, 27, 51}, {28, 24, 45}, {35, 37, 52}, {21, 26, 36}, {16, 19, 24}, {17, 19, 19}, {16, 17, 16}, {17, 17, 16}},
    	    {{20, 20, 21}, {21, 21, 21}, {20, 21, 20}, {20, 20, 21}, {20, 21, 21}, {24, 22, 21}, {37, 34, 41}, {31, 32, 46}, {24, 26, 44}, {39, 33, 49}, {61, 54, 62}, {73, 69, 73}, {71, 70, 71}, {70, 71, 72}, {71, 71, 72}, {71, 71, 72}, {72, 71, 71}, {72, 71, 67}, {73, 70, 71}, {67, 66, 72}, {54, 57, 68}, {30, 36, 53}, {21, 24, 46}, {34, 31, 51}, {34, 34, 51}, {18, 22, 29}, {17, 19, 20}, {18, 19, 19}, {18, 18, 18}, {17, 17, 17}},
    	    {{21, 20, 20}, {20, 19, 19}, {18, 19, 19}, {19, 20, 20}, {19, 21, 21}, {19, 22, 22}, {28, 33, 34}, {32, 39, 44}, {23, 29, 45}, {23, 26, 45}, {36, 33, 49}, {65, 52, 63}, {76, 69, 75}, {70, 69, 72}, {70, 70, 71}, {70, 71, 72}, {69, 71, 71}, {70, 70, 71}, {67, 66, 71}, {53, 58, 69}, {30, 39, 55}, {20, 25, 45}, {25, 27, 48}, {33, 31, 48}, {26, 26, 37}, {18, 19, 23}, {17, 17, 18}, {18, 18, 18}, {18, 17, 17}, {18, 18, 18}},
    	    {{19, 19, 19}, {19, 19, 19}, {20, 20, 20}, {20, 20, 20}, {20, 21, 20}, {19, 21, 20}, {20, 23, 18}, {32, 35, 30}, {37, 40, 51}, {22, 27, 48}, {22, 23, 48}, {31, 26, 44}, {46, 44, 56}, {60, 60, 66}, {64, 64, 67}, {66, 67, 71}, {64, 68, 72}, {56, 59, 64}, {47, 48, 60}, {28, 32, 51}, {18, 23, 46}, {25, 24, 47}, {31, 30, 51}, {28, 29, 41}, {21, 21, 23}, {17, 17, 18}, {18, 18, 18}, {18, 18, 18}, {17, 17, 17}, {17, 17, 17}},
    	    {{19, 19, 19}, {18, 19, 19}, {19, 20, 20}, {21, 21, 20}, {22, 22, 22}, {23, 23, 23}, {22, 22, 20}, {26, 25, 22}, {34, 35, 35}, {31, 37, 45}, {21, 27, 47}, {21, 23, 44}, {20, 22, 43}, {23, 26, 45}, {29, 30, 48}, {31, 33, 51}, {25, 32, 49}, {22, 28, 46}, {21, 24, 41}, {23, 25, 42}, {27, 27, 48}, {34, 30, 48}, {35, 33, 43}, {19, 23, 28}, {18, 19, 20}, {19, 19, 20}, {19, 19, 20}, {20, 20, 21}, {19, 19, 19}, {17, 17, 17}},
    	    {{19, 19, 20}, {19, 20, 20}, {20, 21, 20}, {21, 22, 21}, {22, 23, 22}, {21, 21, 23}, {21, 22, 22}, {21, 21, 20}, {21, 23, 18}, {27, 32, 32}, {30, 32, 45}, {35, 29, 46}, {29, 29, 49}, {21, 25, 46}, {21, 23, 45}, {19, 23, 44}, {17, 27, 48}, {19, 24, 47}, {30, 29, 47}, {36, 37, 51}, {31, 33, 49}, {27, 29, 42}, {23, 24, 28}, {17, 18, 19}, {17, 17, 18}, {18, 17, 18}, {17, 16, 18}, {17, 17, 18}, {17, 17, 18}, {16, 16, 17}},
    	    {{20, 20, 20}, {21, 21, 21}, {21, 21, 20}, {21, 21, 21}, {21, 21, 21}, {20, 20, 21}, {20, 21, 21}, {20, 20, 20}, {18, 19, 19}, {18, 20, 21}, {22, 22, 26}, {30, 25, 28}, {32, 32, 34}, {34, 39, 44}, {35, 39, 47}, {35, 40, 50}, {36, 43, 54}, {35, 39, 50}, {39, 37, 47}, {30, 30, 40}, {21, 23, 29}, {17, 20, 23}, {16, 18, 19}, {18, 18, 19}, {17, 16, 18}, {17, 16, 16}, {17, 16, 16}, {17, 17, 17}, {17, 16, 17}, {17, 16, 17}},
    	    {{20, 20, 20}, {21, 20, 20}, {20, 20, 19}, {20, 20, 20}, {19, 19, 19}, {20, 20, 20}, {20, 21, 21}, {20, 20, 20}, {18, 18, 18}, {19, 18, 19}, {20, 20, 21}, {20, 20, 20}, {19, 20, 19}, {19, 22, 22}, {19, 29, 30}, {20, 29, 33}, {21, 22, 28}, {21, 21, 25}, {20, 19, 24}, {17, 18, 22}, {18, 18, 20}, {18, 18, 18}, {19, 18, 18}, {18, 17, 17}, {17, 16, 17}, {17, 15, 15}, {17, 15, 15}, {17, 15, 15}, {17, 15, 16}, {18, 16, 17}},
    	    {{20, 20, 20}, {20, 20, 20}, {20, 20, 20}, {20, 20, 20}, {18, 18, 18}, {19, 19, 18}, {21, 20, 20}, {20, 20, 19}, {19, 19, 19}, {20, 19, 20}, {19, 19, 20}, {19, 19, 19}, {19, 19, 18}, {20, 20, 19}, {20, 24, 23}, {20, 24, 27}, {19, 20, 22}, {18, 18, 18}, {18, 18, 18}, {17, 17, 17}, {17, 17, 17}, {18, 17, 17}, {18, 17, 17}, {16, 16, 16}, {17, 16, 16}, {17, 15, 15}, {16, 15, 15}, {17, 15, 15}, {17, 14, 15}, {18, 16, 17}},
    	    {{20, 20, 20}, {19, 19, 20}, {20, 20, 20}, {20, 19, 21}, {18, 18, 18}, {18, 18, 18}, {19, 19, 18}, {20, 20, 19}, {20, 20, 20}, {20, 19, 20}, {19, 19, 19}, {19, 19, 19}, {18, 18, 18}, {20, 20, 18}, {25, 25, 24}, {24, 25, 28}, {19, 20, 23}, {18, 18, 19}, {19, 19, 19}, {17, 17, 17}, {17, 17, 17}, {18, 18, 17}, {17, 17, 17}, {16, 17, 17}, {16, 16, 16}, {16, 15, 15}, {16, 15, 16}, {17, 16, 15}, {18, 15, 15}, {18, 16, 17}},
    	    {{19, 19, 19}, {19, 19, 20}, {19, 19, 20}, {18, 18, 19}, {18, 17, 17}, {17, 17, 17}, {18, 18, 17}, {19, 19, 19}, {19, 19, 19}, {18, 18, 18}, {19, 19, 18}, {19, 19, 19}, {19, 19, 19}, {20, 19, 18}, {25, 27, 25}, {24, 27, 30}, {18, 20, 23}, {18, 18, 18}, {19, 19, 19}, {19, 19, 19}, {18, 18, 18}, {19, 18, 17}, {18, 17, 17}, {17, 17, 17}, {17, 16, 16}, {16, 15, 16}, {16, 15, 17}, {17, 16, 16}, {18, 17, 17}, {19, 18, 18}},
    	    {{18, 18, 18}, {19, 19, 19}, {19, 19, 20}, {18, 18, 18}, {19, 18, 18}, {17, 17, 17}, {17, 17, 17}, {17, 17, 18}, {18, 18, 18}, {19, 19, 19}, {19, 19, 20}, {19, 19, 19}, {19, 19, 18}, {20, 20, 18}, {24, 28, 26}, {24, 28, 30}, {19, 21, 23}, {18, 18, 18}, {18, 17, 17}, {19, 19, 19}, {19, 19, 19}, {17, 17, 17}, {18, 17, 17}, {17, 17, 17}, {17, 17, 18}, {17, 17, 18}, {16, 16, 18}, {17, 16, 18}, {17, 16, 17}, {17, 17, 17}}
    	};

    // Feed input into the stream
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 30; j++) {
            for (int c = 0; c < 3; c++) {
                fixed_t val;
                val = test_input[i][j][c];
                input_stream.write(val);
            }
        }
    }

    // Call the CNN forward function
    cnn(input_stream, output_stream);

    return 0;
}
