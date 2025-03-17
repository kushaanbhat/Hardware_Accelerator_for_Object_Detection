#include "cnn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

void cnn(axi_stream &input_stream, axi_stream &output_stream);

std::vector<std::vector<std::vector<float>>> load_test_input(const std::string &filename) {
    std::vector<std::vector<std::vector<float>>> test_input(30, std::vector<std::vector<float>>(30, std::vector<float>(3)));
    std::ifstream file(filename);
    std::string line;
    int i = 0, j = 0, c = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            test_input[i][j][c] = value;
            c++;
            if (c == 3) {
                c = 0;
                j++;
                if (j == 30) {
                    j = 0;
                    i++;
                }
            }
        }
    }

    return test_input;
}

int main() {
    // Declare input and output streams
    axi_stream input_stream;
    axi_stream output_stream;

    // Load test input data from a file
    std::vector<std::vector<std::vector<float>>> test_input = load_test_input("test_input.txt");

    // Feed input into the stream
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 30; j++) {
            for (int c = 0; c < 3; c++) {
                axi_t val;
                val.data = test_input[i][j][c];
                input_stream.write(val);
            }
        }
    }

    // Call the CNN forward function
    cnn(input_stream, output_stream);

    return 0;
}
