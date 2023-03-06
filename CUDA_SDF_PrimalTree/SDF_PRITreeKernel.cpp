#include "SDF_PRITreeKernel.h"

SDF_PRITreeKernel::SDF_PRITreeKernel() {
}
SDF_PRITreeKernel::SDF_PRITreeKernel(KDTree* kdTree) {
	_kdTree = kdTree;
	_kdTree->buildTree();
	_kernel = new SDFKernel(_kdTree->_mesh);
	_root = new PRINode(_kdTree->_root->min(), _kdTree->_root->max() - _kdTree->_root->min(), -1);
	_maxDepth = 8;
}
SDF_PRITreeKernel::~SDF_PRITreeKernel() {
}

void SDF_PRITreeKernel::save(int width, int height)
{
	if (_root->_depth == -1)
		return;
	for (int k = 0; k < 50; k++) {
		char name[30] = "images/PrimalTree/img";
		strcat(name, to_string(k).c_str());
		strcat(name, ".jpg");
		printf("%s save...\n", name);
		clock_t timer = clock();
		Mat img = Mat(width, height, CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				double3 v = make_double3((double)i / (double)(width - 1),
					(double)j / (double)(height - 1), (double)k / 49.0);
				v *= _root->size();
				v += _root->center();
				uchar* c = img.data + (j * img.cols * 3 + i * 3);

				double dist = getDistance(v) * fabs(getDistance(v));
				if (fabs(dist) < 0.0001) {
					c[0] = c[1] = c[2] = 255;
					continue;
				}
				auto node0 = getNearestNode(make_double3(v.x - 0.5 * _root->size().x / (double)width, v.y - 0.5 * _root->size().y / (double)height, v.z));
				auto node1 = getNearestNode(make_double3(v.x + 0.5 * _root->size().x / (double)width, v.y + 0.5 * _root->size().y / (double)height, v.z));
				if (node0 != node1) {
					c[0] = c[1] = c[2] = 0;
					continue;
				}

				if (dist < -0.0) {
					c[0] = c[1] = c[2] = 100;
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
		printf("k = %d, Elapsed = %.0LF msec\n\n", k, (long double)(clock() - timer));
		imwrite(name, img);
	}
	printf("Done!");
}
PRINode* SDF_PRITreeKernel::getNearestNode(double3 p) {
	PRINode* nodes[8];
	PRINode* temp[8];
	PRINode* result = _root;
	double3 point = p;

	if (_root->_depth == -1)
		return NULL;

	p += _root->size() * 0.5;
	p /= _root->size();
	for (int i = 0; i < 8; i++)
		nodes[i] = temp[i] = _root->_childs[i];
	while (1) {
		int px = (p.x >= 0.5);
		int py = (p.y >= 0.5);
		int pz = (p.z >= 0.5);
		int nx = 1 - px;
		int ny = 1 - py;
		int nz = 1 - pz;

		for (int i = 0; i < 8; i++) {
			nodes[i] = (temp[NUM_P[px + py * 2 + pz * 4][i]] ? temp[NUM_P[px + py * 2 + pz * 4][i]]->_childs[NUM_C[px + py * 2 + pz * 4][i]] : nullptr);
			if (nodes[i] && nodes[i]->isInside(point))
				result = nodes[i];
		}
		if (!nodes[0] && !nodes[1] && !nodes[2] && !nodes[3] && !nodes[4] && !nodes[5] && !nodes[6] && !nodes[7])
			return result;

		for (int i = 0; i < 8; i++)
			temp[i] = nodes[i];

		p.x = (p.x - 0.5 * px) * 2;
		p.y = (p.y - 0.5 * py) * 2;
		p.z = (p.z - 0.5 * pz) * 2;
	}
}
double SDF_PRITreeKernel::getDistance(double3 p) {
	PRINode* nodes[8];
	PRINode* temp[8];
	double dists[8];

	if (_root->_depth == -1)
		return NULL;

	p += _root->size() * 0.5;
	p /= _root->size();
	for (int i = 0; i < 8; i++) {
		nodes[i] = temp[i] = _root->_childs[i];
		if (nodes[i])
			dists[i] = nodes[i]->_dist;
	}
	while (1) {
		int px = (p.x >= 0.5);
		int py = (p.y >= 0.5);
		int pz = (p.z >= 0.5);
		int nx = 1 - px;
		int ny = 1 - py;
		int nz = 1 - pz;

		for (int i = 0; i < 8; i++)
			nodes[i] = (temp[NUM_P[px + py * 2 + pz * 4][i]] ? temp[NUM_P[px + py * 2 + pz * 4][i]]->_childs[NUM_C[px + py * 2 + pz * 4][i]] : nullptr);

		if (!nodes[0] && !nodes[1] && !nodes[2] && !nodes[3] && !nodes[4] && !nodes[5] && !nodes[6] && !nodes[7])
			return triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], p);

		dists[nx + ny * 2 + nz * 4] = (nodes[nx + ny * 2 + nz * 4] ? nodes[nx + ny * 2 + nz * 4]->_dist :
			triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7]));

		dists[nx + ny * 2 + pz * 4] = (nodes[nx + ny * 2 + pz * 4] ? nodes[nx + ny * 2 + pz * 4]->_dist :
			biLerp(dists[px + py * 2 + pz * 4], dists[nx + py * 2 + pz * 4], dists[px + ny * 2 + pz * 4], dists[nx + ny * 2 + pz * 4]));
		dists[nx + py * 2 + nz * 4] = (nodes[nx + py * 2 + nz * 4] ? nodes[nx + py * 2 + nz * 4]->_dist :
			biLerp(dists[px + py * 2 + pz * 4], dists[nx + py * 2 + pz * 4], dists[px + py * 2 + nz * 4], dists[nx + py * 2 + nz * 4]));
		dists[px + ny * 2 + nz * 4] = (nodes[px + ny * 2 + nz * 4] ? nodes[px + ny * 2 + nz * 4]->_dist :
			biLerp(dists[px + py * 2 + pz * 4], dists[px + ny * 2 + pz * 4], dists[px + py * 2 + nz * 4], dists[px + ny * 2 + nz * 4]));

		dists[px + ny * 2 + pz * 4] = (nodes[px + ny * 2 + pz * 4] ? nodes[px + ny * 2 + pz * 4]->_dist :
			Lerp(dists[px + ny * 2 + pz * 4], dists[px + py * 2 + pz * 4]));
		dists[nx + py * 2 + pz * 4] = (nodes[nx + py * 2 + pz * 4] ? nodes[nx + py * 2 + pz * 4]->_dist :
			Lerp(dists[nx + py * 2 + pz * 4], dists[px + py * 2 + pz * 4]));
		dists[px + py * 2 + nz * 4] = (nodes[px + py * 2 + nz * 4] ? nodes[px + py * 2 + nz * 4]->_dist :
			Lerp(dists[px + py * 2 + nz * 4], dists[px + py * 2 + pz * 4]));

		for (int i = 0; i < 8; i++)
			temp[i] = nodes[i];

		p.x = (p.x - 0.5 * px) * 2;
		p.y = (p.y - 0.5 * py) * 2;
		p.z = (p.z - 0.5 * pz) * 2;
	}
}
double SDF_PRITreeKernel::getDistWithGrad(double3 p, double3& gradient) {
	PRINode* nodes[8];
	PRINode* temp[8];
	double dists[8];

	if (_root->_depth == -1)
		return NULL;

	p += _root->size() * 0.5;
	p /= _root->size();
	for (int i = 0; i < 8; i++) {
		nodes[i] = temp[i] = _root->_childs[i];
		if (nodes[i])
			dists[i] = nodes[i]->_dist;
	}
	while (1) {
		int px = (p.x >= 0.5);
		int py = (p.y >= 0.5);
		int pz = (p.z >= 0.5);
		int nx = 1 - px;
		int ny = 1 - py;
		int nz = 1 - pz;

		for (int i = 0; i < 8; i++)
			nodes[i] = (temp[NUM_P[px + py * 2 + pz * 4][i]] ? temp[NUM_P[px + py * 2 + pz * 4][i]]->_childs[NUM_C[px + py * 2 + pz * 4][i]] : nullptr);

		if (!nodes[0] && !nodes[1] && !nodes[2] && !nodes[3] && !nodes[4] && !nodes[5] && !nodes[6] && !nodes[7]) {
			gradient.x = biLerp(dists[1], dists[5], dists[3], dists[7], make_double2(p.z, p.y))
				- biLerp(dists[0], dists[4], dists[2], dists[6], make_double2(p.z, p.y));
			gradient.y = biLerp(dists[2], dists[3], dists[6], dists[7], make_double2(p.x, p.z))
				- biLerp(dists[0], dists[1], dists[4], dists[5], make_double2(p.x, p.z));
			gradient.z = biLerp(dists[4], dists[5], dists[6], dists[7], make_double2(p.x, p.y))
				- biLerp(dists[0], dists[1], dists[2], dists[3], make_double2(p.x, p.y));
			Normalize(gradient);
			return triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], p);
		}

		dists[nx + ny * 2 + nz * 4] = (nodes[nx + ny * 2 + nz * 4] ? nodes[nx + ny * 2 + nz * 4]->_dist :
			triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7]));

		dists[nx + ny * 2 + pz * 4] = (nodes[nx + ny * 2 + pz * 4] ? nodes[nx + ny * 2 + pz * 4]->_dist :
			biLerp(dists[px + py * 2 + pz * 4], dists[nx + py * 2 + pz * 4], dists[px + ny * 2 + pz * 4], dists[nx + ny * 2 + pz * 4]));
		dists[nx + py * 2 + nz * 4] = (nodes[nx + py * 2 + nz * 4] ? nodes[nx + py * 2 + nz * 4]->_dist :
			biLerp(dists[px + py * 2 + pz * 4], dists[nx + py * 2 + pz * 4], dists[px + py * 2 + nz * 4], dists[nx + py * 2 + nz * 4]));
		dists[px + ny * 2 + nz * 4] = (nodes[px + ny * 2 + nz * 4] ? nodes[px + ny * 2 + nz * 4]->_dist :
			biLerp(dists[px + py * 2 + pz * 4], dists[px + ny * 2 + pz * 4], dists[px + py * 2 + nz * 4], dists[px + ny * 2 + nz * 4]));

		dists[px + ny * 2 + pz * 4] = (nodes[px + ny * 2 + pz * 4] ? nodes[px + ny * 2 + pz * 4]->_dist :
			Lerp(dists[px + ny * 2 + pz * 4], dists[px + py * 2 + pz * 4]));
		dists[nx + py * 2 + pz * 4] = (nodes[nx + py * 2 + pz * 4] ? nodes[nx + py * 2 + pz * 4]->_dist :
			Lerp(dists[nx + py * 2 + pz * 4], dists[px + py * 2 + pz * 4]));
		dists[px + py * 2 + nz * 4] = (nodes[px + py * 2 + nz * 4] ? nodes[px + py * 2 + nz * 4]->_dist :
			Lerp(dists[px + py * 2 + nz * 4], dists[px + py * 2 + pz * 4]));

		for (int i = 0; i < 8; i++)
			temp[i] = nodes[i];

		p.x = (p.x - 0.5 * px) * 2;
		p.y = (p.y - 0.5 * py) * 2;
		p.z = (p.z - 0.5 * pz) * 2;
	}
}
bool SDF_PRITreeKernel::comparison(PRINode* node, double* cdists) {
	double dists[7];
	if (fabs(Lerp(node->_dist, dists[0] = _kdTree->getDistToPoint(node->getCorner(8))) - cdists[1]) > _error) //x
		return true;
	if (fabs(Lerp(node->_dist, dists[1] = _kdTree->getDistToPoint(node->getCorner(9))) - cdists[2]) > _error) //y
		return true;
	if (fabs(Lerp(node->_dist, dists[3] = _kdTree->getDistToPoint(node->getCorner(11))) - cdists[4]) > _error) //z
		return true;
	if (fabs(biLerp(node->_dist, dists[0], dists[1], dists[2] = _kdTree->getDistToPoint(node->getCorner(10))) - cdists[3]) > _error) //xy
		return true;
	if (fabs(biLerp(node->_dist, dists[0], dists[3], dists[4] = _kdTree->getDistToPoint(node->getCorner(12))) - cdists[5]) > _error) //xz
		return true;
	if (fabs(biLerp(node->_dist, dists[1], dists[3], dists[5] = _kdTree->getDistToPoint(node->getCorner(13))) - cdists[6]) > _error) //yz
		return true;
	if (fabs(triLerp(node->_dist, dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6] = _kdTree->getDistToPoint(node->getCorner(14))) - cdists[7]) > _error) //center
		return true;
	for (int i = 0; i < 3; i++) {
		int a = 1 + i * 2;
		if (fabs(Lerp(dists[a], dists[a + 1]) - _kdTree->getDistToPoint((node->getCorner(a + 8) + node->getCorner(a + 9)) / 2)) > _error) // x-Lines
			return true;
		a = (3 + i) % 5;
		if (fabs(Lerp(dists[a], dists[a + 2]) - _kdTree->getDistToPoint((node->getCorner(a + 8) + node->getCorner(a + 10)) / 2)) > _error) // y-Lines
			return true;
		if (fabs(Lerp(dists[i], dists[i + 4]) - _kdTree->getDistToPoint((node->getCorner(i + 8) + node->getCorner(i + 12)) / 2)) > _error) // z-Lines
			return true;
	}
	if (fabs(biLerp(dists[0], dists[2], dists[4], dists[6]) - _kdTree->getDistToPoint((node->getCorner(8) + node->getCorner(14)) / 2)) > _error) //opposite yz
		return true;
	if (fabs(biLerp(dists[1], dists[2], dists[5], dists[6]) - _kdTree->getDistToPoint((node->getCorner(9) + node->getCorner(14)) / 2)) > _error) // opposite xz
		return true;
	if (fabs(biLerp(dists[3], dists[4], dists[5], dists[6]) - _kdTree->getDistToPoint((node->getCorner(11) + node->getCorner(14)) / 2)) > _error) // opposite xy
		return true;
	return false;
}
void SDF_PRITreeKernel::getDistToPointsKernel(vector<double3>& h_points, double3* d_points, vector<double>& h_dists, double* d_dists) {
	int oldNum = h_dists.size();
	h_dists.resize(oldNum + h_points.size());
	cudaMemcpyAsync(d_points, &h_points[0], h_points.size() * sizeof(double3), cudaMemcpyHostToDevice, 0);
	_kernel->getDistToPoints(d_points, h_points.size(), d_dists);
	cudaMemcpyAsync(&h_dists[oldNum], d_dists, h_points.size() * sizeof(double), cudaMemcpyDeviceToHost, 0);
	h_points.clear();
}
void SDF_PRITreeKernel::buildTree(void) {
	if (!_root)
		return;
	_root->_depth = 0;
	_root->_dist = ssqrt(_kdTree->getDistToPoint(_root->center()));

	printf("SDFPRIKernel build...\n");
	clock_t timer = clock();
	vector<double3> h_points;
	vector<double3*> d_points;
	vector<double> h_dists;
	vector<double*> d_dists;
	for (int i = 1; i < 8; i++)
		h_points.push_back(_root->getCorner(i));
	double3* d_p;
	double* d_d;
	CUDA_CHECK(cudaMalloc((void**)&d_p, h_points.size() * sizeof(double3)));
	CUDA_CHECK(cudaMalloc((void**)&d_d, h_points.size() * sizeof(double)));
	getDistToPointsKernel(h_points, d_p, h_dists, d_d);
	cudaFree(d_p);
	cudaFree(d_d);
	
	int oldDepth = -1;
	int pointIndex = 0;
	int distIndex = 0;

	vector<PRINode*> queue;
	queue.push_back(_root);
	while (queue.size()) {
		auto node = queue[0];
		queue.erase(queue.begin());
		if (node->_depth >= _maxDepth)
			break;
		double dists[8];
		uint b_subDiv = 0;
		uint subIndex = 1;
		if (node->_depth != oldDepth) {
			for (int i = 0; i < d_points.size(); i++) {
				CUDA_CHECK(cudaFree(d_points[i]));
				CUDA_CHECK(cudaFree(d_dists[i]));
			}
			d_points.clear();
			d_dists.clear();
			oldDepth = node->_depth;
		}
		if (h_points.size()) {
			CUDA_CHECK(cudaMalloc((void**)&d_p, h_points.size() * sizeof(double3)));
			CUDA_CHECK(cudaMalloc((void**)&d_d, h_points.size() * sizeof(double)));
			d_points.push_back(d_p);
			d_dists.push_back(d_d);
			getDistToPointsKernel(h_points, d_p, h_dists, d_d);
		}
		double crossSize = Length(node->size());

		if (fabs(dists[0] = node->_dist) <= crossSize)
			b_subDiv ^= subIndex;
		subIndex += subIndex;
		for (int i = 1; i < 8; i++) {
			if (fabs(dists[i] = ssqrt(h_dists[distIndex++])) <= crossSize)
				b_subDiv ^= subIndex;
			subIndex += subIndex;
		}
		if (b_subDiv)
			if (!comparison(node, dists))
				b_subDiv = 0;

		if (!b_subDiv)
			continue;
		subIndex = 1;
		for (int i = 0; i < 8; i++) {
			if (b_subDiv & subIndex) {
				node->subdivide(i, dists[i], _root->center() + _root->size());
				if (node->_childs[i]) {
					if(node->_childs[i]->_depth < _maxDepth)
						for (int j = 1; j < 8; j++)
							h_points.push_back(node->_childs[i]->getCorner(j));
					queue.push_back(node->_childs[i]);
				}
			}
			subIndex += subIndex;
		}
	}
	h_dists.clear();
	printf("Done! Elapsed: %.3lf msec\n", (long double)(clock() - timer));
}