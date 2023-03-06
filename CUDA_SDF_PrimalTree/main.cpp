#include "Viewer.h"
//#include "SDF_PRITree.h"
#include "SDF_PRITreeKernel.h"
#include "SDF_PRITreeKDKernel.h"

void saveImage(double* v, int _res) {
	for (int k = 0; k < 50; k++) {
		char name[50] = "images/img";
		strcat(name, to_string(k).c_str());
		strcat(name, ".jpg");
		Mat img = Mat(_res, _res, CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i < _res; i++) {
			for (int j = 0; j < _res; j++) {
				uchar* c = img.data + (j * img.cols * 3 + i * 3);

				auto dist = v[i + j * _res + ((k * _res) / 50) * _res * _res];
				if (dist < -0.0001) {
					c[0] = c[1] = c[2] = 100;
					continue;
				}
				if (dist < 0.0001) {
					c[0] = c[1] = c[2] = 255;
					continue;
				}
				dist *= 6.0;
				int integer = (int)dist;
				double frac = dist - (double)integer;

				if (frac <= 0.25) {
					frac = (frac / 0.25);
				}
				else if (frac <= 0.5) {
					frac -= 0.25;
					frac = 1.0 - (frac / 0.25);
				}
				else if (frac <= 0.75) {
					frac -= 0.5;
					frac = (frac / 0.25);
				}
				else if (frac <= 1) {
					frac -= 0.75;
					frac = 1 - (frac / 0.25);
				}
				double3 rainbow = SCALAR_TO_COLOR(frac) * 255.0;

				c[0] = rainbow.x;
				c[1] = rainbow.y;
				c[2] = rainbow.z;
			}
		}
		imwrite(name, img);
	}
}
int main(int argc, char** argv) {
	/*Viewer* v = new Viewer();
	v->init(v);
	v->show(&argc, argv);*/

	Mesh* _mesh = new Mesh("../../include/obj/bunny2.obj", 0.6);
	KDTree* _kdTree = new KDTree(_mesh);
	SDF_PRITree* h_tree = new SDF_PRITree(_kdTree);
	h_tree->buildTree();
	h_tree->clear();
	h_tree->buildTreeKernel();
	h_tree->save(512, 512);

	return 0;
}