#include "SDF_PRITree.h"

SDF_PRITree::SDF_PRITree() {
	_root = nullptr;
}
SDF_PRITree::SDF_PRITree(KDTree* kdTree) {
	_kdTree = kdTree;
	_kdTree->buildTree();
	_kernel = new SDFKernel();
	_kernel->copyGPU(kdTree->_mesh);
	_root = new PRINode(_kdTree->_root->min(), _kdTree->_root->max() - _kdTree->_root->min(), -1);
	_maxDepth = 8;
}
SDF_PRITree::~SDF_PRITree() {
}

void SDF_PRITree::clearUp(PRINode* node) {
	for (int i = 0; i < 2; i++)
		if (node->_childs[i])
			clearUp(node->_childs[i]);
	node->clear();
}
void SDF_PRITree::clear(void) {
	if (_root->_depth != -1)
		clearUp(_root);
	_root = new PRINode(_kdTree->_root->min(), _kdTree->_root->max() - _kdTree->_root->min(), -1);
}
void SDF_PRITree::drawUp(PRINode* node) {
	for (int i = 0; i < 2; i++)
		if (node->_childs[i])
			drawUp(node->_childs[i]);
	node->draw();
}
void SDF_PRITree::draw(void) {
	if (_root->_depth != -1)
		drawUp(_root);
}
void SDF_PRITree::drawBoundary(void) {
	glPushMatrix();
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glTranslated(0.0, 0.0, _root->center().z);
	glBegin(GL_LINES);
	glVertex3d(_root->center().x, _root->center().y, 0.0);
	glVertex3d(_root->center().x, _root->center().y, _root->size().z);
	glVertex3d(_root->center().x, _root->center().y + _root->size().y, 0.0);
	glVertex3d(_root->center().x, _root->center().y + _root->size().y, _root->size().z);
	glVertex3d(_root->center().x + _root->size().x, _root->center().y, 0.0);
	glVertex3d(_root->center().x + _root->size().x, _root->center().y, _root->size().y);
	glVertex3d(_root->center().x + _root->size().x, _root->center().y + _root->size().y, 0.0);
	glVertex3d(_root->center().x + _root->size().x, _root->center().y + _root->size().y, _root->size().z);
	glEnd();

	glRectd(_root->center().x, _root->center().y, _root->center().x + _root->size().x, _root->center().y + _root->size().y);
	glTranslated(0.0, 0.0, _root->size().z);
	glRectd(_root->center().x, _root->center().y, _root->center().x + _root->size().x, _root->center().y + _root->size().y);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPopMatrix();
}
void SDF_PRITree::save(int width, int height)
{
	if (_root->_depth == -1)
		return;
	for (int k = 0; k < 50; k++) {
		char name[17] = "images/PRI/img";
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
PRINode* SDF_PRITree::getNearestNode(double3 p) {
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
double SDF_PRITree::getDistance(double3 p) {
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
double SDF_PRITree::getDistWithGrad(double3 p, double3& gradient) {
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
bool SDF_PRITree::comparison(PRINode* node, double* cdists) {
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
void SDF_PRITree::buildTree(void) {
	if (!_root)
		return;
	_root->_depth = 0;
	_root->_dist = ssqrt(_kdTree->getDistToPoint(_root->center()));
	vector<PRINode*> queue;
	queue.push_back(_root);
	printf("PRI build CPU KDTree...\n");
	clock_t timer = clock();
	while (queue.size() != 0) {
		auto node = queue[0];
		queue.erase(queue.begin());
		if (node->_depth >= _maxDepth)
			break;
		double crossSize = Length(node->size());
		double dists[8];
		bool b_subDiv[8] = { false };

		dists[0] = node->_dist;
		for (int i = 1; i < 8; i++)
			dists[i] = ssqrt(_kdTree->getDistToPoint(node->getCorner(i)));

		for (int i = 0; i < 8; i++)
			if (fabs(dists[i]) <= crossSize)
				b_subDiv[i] = true;

		if (b_subDiv[0] || b_subDiv[1] || b_subDiv[2] || b_subDiv[3] ||
			b_subDiv[4] || b_subDiv[5] || b_subDiv[6] || b_subDiv[7])
			if (!comparison(node, dists))
				b_subDiv[0] = b_subDiv[1] = b_subDiv[2] = b_subDiv[3] =
				b_subDiv[4] = b_subDiv[5] = b_subDiv[6] = b_subDiv[7] = false;

		for (int i = 0; i < 8; i++)
			if (b_subDiv[i]) {
				node->subdivide(i, dists[i], _root->center() + _root->size());
				if (node->_childs[i])
					queue.push_back(node->_childs[i]);
			}
	}
	printf("Done! Elapsed: %.3lf msec\n", (long double)(clock() - timer));
}
bool SDF_PRITree::comparisonKernel(PRINode* node, double* cdists) {
	double dists[7];
	if (fabs(Lerp(node->_dist, dists[0] = _kernel->getDistToPoint(node->getCorner(8))) - cdists[1]) > _error) //x
		return true;
	if (fabs(Lerp(node->_dist, dists[1] = _kernel->getDistToPoint(node->getCorner(9))) - cdists[2]) > _error) //y
		return true;
	if (fabs(Lerp(node->_dist, dists[3] = _kernel->getDistToPoint(node->getCorner(11))) - cdists[4]) > _error) //z
		return true;
	if (fabs(biLerp(node->_dist, dists[0], dists[1], dists[2] = _kernel->getDistToPoint(node->getCorner(10))) - cdists[3]) > _error) //xy
		return true;
	if (fabs(biLerp(node->_dist, dists[0], dists[3], dists[4] = _kernel->getDistToPoint(node->getCorner(12))) - cdists[5]) > _error) //xz
		return true;
	if (fabs(biLerp(node->_dist, dists[1], dists[3], dists[5] = _kernel->getDistToPoint(node->getCorner(13))) - cdists[6]) > _error) //yz
		return true;
	if (fabs(triLerp(node->_dist, dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6] = _kernel->getDistToPoint(node->getCorner(14))) - cdists[7]) > _error) //center
		return true;
	for (int i = 0; i < 3; i++) {
		int a = 1 + i * 2;
		if (fabs(Lerp(dists[a], dists[a + 1]) - _kernel->getDistToPoint((node->getCorner(a + 8) + node->getCorner(a + 9)) / 2)) > _error) // x-Lines
			return true;
		a = (3 + i) % 5;
		if (fabs(Lerp(dists[a], dists[a + 2]) - _kernel->getDistToPoint((node->getCorner(a + 8) + node->getCorner(a + 10)) / 2)) > _error) // y-Lines
			return true;
		if (fabs(Lerp(dists[i], dists[i + 4]) - _kernel->getDistToPoint((node->getCorner(i + 8) + node->getCorner(i + 12)) / 2)) > _error) // z-Lines
			return true;
	}
	if (fabs(biLerp(dists[0], dists[2], dists[4], dists[6]) - _kernel->getDistToPoint((node->getCorner(8) + node->getCorner(14)) / 2)) > _error) //opposite yz
		return true;
	if (fabs(biLerp(dists[1], dists[2], dists[5], dists[6]) - _kernel->getDistToPoint((node->getCorner(9) + node->getCorner(14)) / 2)) > _error) // opposite xz
		return true;
	if (fabs(biLerp(dists[3], dists[4], dists[5], dists[6]) - _kernel->getDistToPoint((node->getCorner(11) + node->getCorner(14)) / 2)) > _error) // opposite xy
		return true;
	return false;
}
void SDF_PRITree::buildTreeKernel(void) {
	if (!_root)
		return;
	_root->_depth = 0;
	_root->_dist = ssqrt(_kernel->getDistToPoint(_root->center()));
	vector<PRINode*> queue;
	queue.push_back(_root);
	printf("PRI build GPU Kernel...\n");
	clock_t timer = clock();
	while (queue.size() != 0) {
		PRINode* node = queue[0];
		//double crossSize = LengthSquared(node->size());
		queue.erase(queue.begin());
		if (node->_depth >= _maxDepth)
			break;
		double crossSize = Length(node->size());
		double dists[8];
		bool b_subDiv[8] = { false };

		dists[0] = node->_dist;
		for (int i = 1; i < 8; i++)
			dists[i] = ssqrt(_kernel->getDistToPoint(node->getCorner(i)));

		for (int i = 0; i < 8; i++)
			if (fabs(dists[i]) <= crossSize)
				b_subDiv[i] = true;

		if (b_subDiv[0] || b_subDiv[1] || b_subDiv[2] || b_subDiv[3] ||
			b_subDiv[4] || b_subDiv[5] || b_subDiv[6] || b_subDiv[7])
			if (!comparisonKernel(node, dists))
				b_subDiv[0] = b_subDiv[1] = b_subDiv[2] = b_subDiv[3] =
				b_subDiv[4] = b_subDiv[5] = b_subDiv[6] = b_subDiv[7] = false;

		for (int i = 0; i < 8; i++)
			if (b_subDiv[i]) {
				node->subdivide(i, dists[i], _root->center() + _root->size());
				if (node->_childs[i])
					queue.push_back(node->_childs[i]);
			}
	}
	printf("Done! Elapsed: %.3lf msec\n", (long double)(clock() - timer));
}