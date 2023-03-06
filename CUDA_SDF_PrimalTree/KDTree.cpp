#include "KDTree.h"

KDTree::KDTree() {
	_mesh = nullptr;
	_root = new KDNode();
	_maxDepth = 10;
	_numNode = 0;
}
KDTree::KDTree(Mesh* m) {
	_mesh = m;
	_root = new KDNode(m->_minBoundary, m->_maxBoundary, -1);
	_maxDepth = 10;
	_numNode = 0;
}
KDTree::~KDTree() {
}
void KDTree::clearUp(KDNode* node) {
	for (int i = 0; i < 2; i++)
		if (node->_childs[i])
			clearUp(node->_childs[i]);
	node->clear();
}
void KDTree::clear(void) {
	if (_root->_depth != -1)
		clearUp(_root);
	_numNode = 0;
}
void KDTree::drawUp(KDNode* node) {
	for (int i = 0; i < 2; i++)
		if (node->_childs[i])
			drawUp(node->_childs[i]);
	node->draw();
}
void KDTree::draw(void) {
	if (_root->_depth != -1)
		drawUp(_root);
}
void KDTree::drawBoundary(void) {
	glPushMatrix();
	glColor3f(1.0f, 1.0f, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_LINES);
	glVertex3d(_root->min().x, _root->min().y, _root->min().z);
	glVertex3d(_root->min().x, _root->min().y, _root->max().z);
	glVertex3d(_root->min().x, _root->max().y, _root->min().z);
	glVertex3d(_root->min().x, _root->max().y, _root->max().z);
	glVertex3d(_root->max().x, _root->min().y, _root->min().z);
	glVertex3d(_root->max().x, _root->min().y, _root->max().z);
	glVertex3d(_root->max().x, _root->max().y, _root->min().z);
	glVertex3d(_root->max().x, _root->max().y, _root->max().z);
	glEnd();
	glTranslated(0, 0, _root->min().z);
	glRectd(_root->min().x, _root->min().y, _root->max().x, _root->max().y);
	glTranslated(0, 0, _root->max().z - _root->min().z);
	glRectd(_root->min().x, _root->min().y, _root->max().x, _root->max().y);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(1.0f, 1.0f, 1.0f);
	glPopMatrix();
}
void KDTree::setMesh(Mesh* m) {
	_mesh = m;
	_root->setMin(m->_minBoundary);
	_root->setMax(m->_maxBoundary);
}
KDNode* KDTree::getNearestNode(double3 p) {
	vector<KDNode*> queue;
	queue.push_back(_root);
	while (queue.size()) {
		KDNode* node = queue[0];
		queue.erase(queue.begin());
		if (!node->_childs[0])
			return node;
		//if (np < node->_min && np > node->_max)
		if (*((double*)&p + node->_divAxis) <= node->_divPos)
			queue.push_back(node->_childs[0]);
		else
			queue.push_back(node->_childs[1]);
	}
	return _root;
}
void KDTree::getDistToPoints(vector<double3> points, double *output) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	for (int i = 0; i < points.size(); i++)
		output[i] = getDistToPoint(points[i]);
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf(" Done!!\nElapsed TIme: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
double KDTree::getDistToPoint(double3 p) {
#ifdef TESTVIEWER
	_testNodes.clear();
	int computeNum = 0;
#endif
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
	printf("Compute KDTree Distance...");
#endif
	KDNode* node = getNearestNode(p);
	double range = DBL_MAX;
	for (auto f : node->_polygons) {
		for (auto v : f->_vertices) {
			double dist = LengthSquared(v->_pos - p);
			if (dist < range)
				range = dist;
		}
	}
#ifdef TESTVIEWER
	_testNodes.push_back(node);
#endif
	vector<KDNode*> queue;
	queue.push_back(_root);
	double minDist = DBL_MAX;
	while (queue.size()) {
		node = queue[0];
		queue.erase(queue.begin());
		if (isIntersectNode(node, p, range)) {
			for (int i = 0; i < node->_polygons.size(); i++) {
#ifdef TESTVIEWER
				if (i == 0 && _testNodes[0] != node)_testNodes.push_back(node);
				computeNum++;
#endif
				double dist = getDistToPoint(p, node->_polygons[i]);
				if (fabs(dist) < fabs(minDist))
					minDist = dist;
			}
			if (node->_childs[0] != nullptr)
				for (int i = 0; i < 2; i++)
					queue.push_back(node->_childs[i]);
		}
	}
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf(" Done!!\nElapsed TIme: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
#ifdef TESTVIEWER
	printf(" Node Num: %d\n", _testNodes.size());
	printf(" Compute Num: %d\n", computeNum);
#endif
	return minDist;
}
double KDTree::getDistToPoint(double3 p, Face* f) {
	double v0pDotv01 = (p - f->_vertices[0]->_pos) * (f->_vertices[1]->_pos - f->_vertices[0]->_pos);
	double v0pDotv02 = (p - f->_vertices[0]->_pos) * (f->_vertices[2]->_pos - f->_vertices[0]->_pos);
	double v01Dotv01 = LengthSquared(f->_vertices[1]->_pos - f->_vertices[0]->_pos);
	double v02Dotv02 = LengthSquared(f->_vertices[2]->_pos - f->_vertices[0]->_pos);
	double v01Dotv02 = (f->_vertices[1]->_pos - f->_vertices[0]->_pos) * (f->_vertices[2]->_pos - f->_vertices[0]->_pos);
	double v1pDotv12 = v0pDotv02 - v0pDotv01 - v01Dotv02 + v01Dotv01;
	double result = 0.0;
	uchar term0 = v0pDotv01 <= 0;
	uchar term1 = v01Dotv01 - v0pDotv01 <= 0;
	uchar term2 = v0pDotv01 - v0pDotv02 - v01Dotv02 + v02Dotv02 <= 0;
	double3 N;
	if (term0 && v0pDotv02 <= 0) {
		p -= f->_vertices[0]->_pos;
		N = f->_vertices[0]->_normal;
	}
	else if (v1pDotv12 <= 0 && term1) {
		p -= f->_vertices[1]->_pos;
		N = f->_vertices[1]->_normal;
	}
	else if (v02Dotv02 - v0pDotv02 <= 0 && term2) {
		p -= f->_vertices[2]->_pos;
		N = f->_vertices[2]->_normal;
	}
	else if (v0pDotv01 * v01Dotv02 - v0pDotv02 * v01Dotv01 >= 0 && !term0 && !term1) {
		p -= f->_vertices[0]->_pos;
		result -= v0pDotv01 * (v0pDotv01 / v01Dotv01);
		N = f->getEdgeNormal(0);
	}
	else if ((v0pDotv01 - v01Dotv01) * (v02Dotv02 - v01Dotv02) - (v0pDotv02 - v01Dotv02) * (v01Dotv02 - v01Dotv01) >= 0 && !term2) {
		p -= f->_vertices[1]->_pos;
		result -= v1pDotv12 * v1pDotv12 / (v01Dotv01 + v02Dotv02 - v01Dotv02 - v01Dotv02);
		N = f->getEdgeNormal(1);
	}
	else if (v0pDotv02 * v01Dotv02 - v0pDotv01 * v02Dotv02 >= 0) {
		p -= f->_vertices[0]->_pos;
		result -= v0pDotv02 * (v0pDotv02 / v02Dotv02);
		N = f->getEdgeNormal(2);
	}
	else {
		result = f->_normal * (p - f->_vertices[0]->_pos);
		return  fabs(result) * result;
	}
	result += p * p;
	if (N * p < 0)
		return -result;
	return result;
}
bool KDTree::isIntersectNode(KDNode* node, double3 p, double range)
{
	double minDist = 0.0;
	for (int i = 0; i < 3; i++) {
		double tmp = *((double*)&p + i) - *((double*)&node->min() + i);
		if (tmp < 0)
			minDist += tmp * tmp;
		else if ((tmp = *((double*)&p + i) - *((double*)&node->max() + i)) > 0) 
			minDist += tmp * tmp;
	}

	return (minDist <= range);
}
void KDTree::subdivide(KDNode* node) {
	double maxDist = DBL_MIN;
	for (int i = 0; i < 3; i++) {
		double dist = *((double*)&node->max() + i) - *((double*)&node->min() + i);
		if (maxDist < dist) {
			maxDist = dist;
			node->_divAxis = i;
		}
	}
	vector<double> minVertices;
	vector<double> maxVertices;
	vector<double> queue;
	uint dAxis = node->_divAxis;
	for (auto f : node->_polygons) {
		double minV, maxV;
		minV = maxV = *((double*)&f->_vertices[0]->_pos + dAxis);
		for (int i = 1; i < 3; i++) {
			double v0 = *((double*)&f->_vertices[i]->_pos + dAxis);
			if (minV > v0)		minV = v0;
			else if (maxV < v0)	maxV = v0;
		}
		minVertices.push_back(minV);
		maxVertices.push_back(maxV);
		queue.push_back(minV);
		queue.push_back(maxV);
	}
	sort(queue.begin(), queue.end(),
		[](double first, double second) -> bool { return first > second; });
	node->_divPos = (queue[(queue.size() - 1u) / 2u] + queue[((queue.size() - 1u) / 2u) + 1u]) * 0.5;
	
	double3 newMin = node->min();
	double3 newMax = node->max();
	*((double*)&newMin + dAxis) = *((double*)&newMax + dAxis) = node->_divPos;
	node->_childs[0] = new KDNode(node->min(), newMax, node->_depth + 1);
	node->_childs[1] = new KDNode(newMin, node->max(), node->_depth + 1);
	for (int i = 0; i < node->_polygons.size(); i++) {
		if (maxVertices[i] < node->_divPos)
			node->_childs[0]->_polygons.push_back(node->_polygons[i]);
		else if (minVertices[i] > node->_divPos)
			node->_childs[1]->_polygons.push_back(node->_polygons[i]);
		else {
			node->_childs[0]->_polygons.push_back(node->_polygons[i]);
			node->_childs[1]->_polygons.push_back(node->_polygons[i]);
		}
	}
	node->_polygons.clear();
	_numNode += 2;
	//printf("%d, %d\n", node->_childs[0]->_polygons.size(), node->_childs[1]->_polygons.size());
}
void KDTree::buildTree(void) {
	if (!_mesh)
		return;
	ctimer timer = CNOW;
	printf("CPU KDTree build...");

	_root->_depth = 0;
	_root->_polygons.clear();
	for (auto f : _mesh->_faces) {
		double3 v0 = f->_vertices[0]->_pos;
		double3 v1 = f->_vertices[1]->_pos;
		double3 v2 = f->_vertices[2]->_pos;
		double3 min = minVec(minVec(v0, v1), v2);
		double3 max = maxVec(maxVec(v0, v1), v2);
		if (min >= _root->min() && max <= _root->max())
			_root->_polygons.push_back(f);
	}
	vector<KDNode*> queue;
	queue.push_back(_root);
	while (queue.size()) {
		KDNode* node = queue[0];
		queue.erase(queue.begin());
		if (node->_depth >= _maxDepth ||
			node->_polygons.size() * 500 <= _mesh->_faces.size())
			continue;

		subdivide(node);
		for (int i = 0; i < 2; i++)
			queue.push_back(node->_childs[i]);
	}
	printf(" Done!!\nElapsed TIme: %lf msec\n", (CNOW - timer) / 10000.0);
}
void KDTree::buildTree(double3 min, double3 max) {
	if (!_mesh)
		return;

	ctimer timer = CNOW;
	printf("CPU KDTree build...");

	_root->setMin(min);
	_root->setMax(max);
	_root->_depth = 0;
	_root->_polygons.clear();
	for (auto f : _mesh->_faces) {
		double3 v0 = f->_vertices[0]->_pos;
		double3 v1 = f->_vertices[1]->_pos;
		double3 v2 = f->_vertices[2]->_pos;
		double3 min = minVec(minVec(v0, v1), v2);
		double3 max = maxVec(maxVec(v0, v1), v2);
		if (min >= _root->min() && max <= _root->max())
			_root->_polygons.push_back(f);
	}
	_numNode = 1;
	vector<KDNode*> queue;
	queue.push_back(_root);
	while (queue.size()) {
		KDNode* node = queue[0];
		queue.erase(queue.begin());
		if (node->_depth >= _maxDepth ||
			node->_polygons.size() * 500 <= _mesh->_faces.size())
			continue;

		subdivide(node);
		for (int i = 0; i < 2; i++)
			queue.push_back(node->_childs[i]);
	}
	printf(" Done!!\nElapsed TIme: %lf msec\n", (CNOW - timer) / 10000.0);
}