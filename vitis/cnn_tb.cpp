#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <iostream>
#include "dataTypes.h"
#include "cnn_tb.h"
#include <iomanip>

using namespace std;

typedef ap_axiu<32, 1, 1, 1> AXI_VAL;

void cnn(hls::stream<AXI_VAL> &image_in, hls::stream<AXI_VAL> &probability_out, hls::stream<AXI_VAL> &class_out);

int main() {
    hls::stream<AXI_VAL> image_in, probability_out, class_out;
    int h = d_out_h;
    //int w = d_out_h;
    //int d = m_out_d;
    float result_class;
    float result[h];
    //float result[h][w];
    //float result[h][w][d];

	for (int i = 0; i < img_h; i++) {
		for (int j = 0; j < img_w; j++) {
			for (int k = 0; k < img_d; k++) {
				image_in.write(float_to_axi(image[i][j][k]));
			}
		}
	}

	cnn(image_in, probability_out, class_out);

	for (int i = 0; i < h; i++) {
		//for (int j = 0; j < w; j++) {
			//for (int k = 0; k < d; k++) {
				AXI_VAL temp = probability_out.read();
				result[i] = axi_to_float(temp);
				//result[i][j][k] = axi_to_float(temp);
			//}
		//}
	}
	AXI_VAL tempa = class_out.read();
	result_class = axi_to_float(tempa);

	cout<<"prediction: "<<result_class<<endl;

	int id = 1;
	for (int i = 0; i < h; i++) {
		//for (int j = 0; j < w; j++) {
			//for (int k = 0; k < d; k++) {
		if (id<10)
		{
			cout<<"[0"<< id<<"] ["<<result[i]<<"] [Class: 0"<<(id-1)<<"]"<<endl;
		}
		if (id == 10)
		{
			cout<<"["<< id<<"] ["<<result[i]<<"] [Class: 0"<<(id-1)<<"]"<<endl;
		}
		if (id>10)
		{
			cout<<"["<< id<<"] ["<<result[i]<<"] [Class: "<<(id-1)<<"]"<<endl;
		}
			//cout<<"id = "<< id<<" "<<result[i][j][k]<<endl;
			id++;
			//}
		//}
	}

    return 0;
}
