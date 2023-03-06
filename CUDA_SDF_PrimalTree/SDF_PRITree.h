#ifndef __SDF_PRIMAL_TREE_H__
#define __SDF_PRIMAL_TREE_H__

#pragma once
#include "KDTree.h"
#include "SDFKernel.h"

using namespace cv;

const unsigned int NUM_P[8][8] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 1, 0, 1, 0, 1, 0, 1 },
	{ 0, 0, 2, 2, 0, 0, 2, 2 },
	{ 0, 1, 2, 3, 0, 1, 2, 3 },
	{ 0, 0, 0, 0, 4, 4, 4, 4 },
	{ 0, 1, 0, 1, 4, 5, 4, 5 },
	{ 0, 0, 2, 2, 4, 4, 6, 6 },
	{ 0, 1, 2, 3, 4, 5, 6, 7 } };
const unsigned int NUM_C[8][8] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7 },
	{ 1, 0, 3, 2, 5, 4, 7, 6 },
	{ 2, 3, 0, 1, 6, 7, 4, 5},
	{ 3, 2, 1, 0, 7, 6, 5, 4 },
	{ 4, 5, 6, 7, 0, 1, 2, 3 },
	{ 5, 4, 7, 6, 1, 0, 3, 2 },
	{ 6, 7, 4, 5, 2, 3, 0, 1 },
	{ 7, 6, 5, 4, 3, 2, 1, 0 } };

class PRINode {
public:
	PRINode** _childs;
	double3* _pos;
	double _dist;
	int _depth;
public:
	PRINode() {
		_childs = new PRINode * [8];
		_pos = new double3[2];
		for (int i = 0; i < 8; i++) {
			_childs[i] = nullptr;
			if (i < 2)
				_pos[i] = make_double3(0.0);
		}
		_dist = 0.0;
		_depth = -1;
	}
	PRINode(double3 min, double3 size, int depth) {
		_childs = new PRINode * [8];
		_pos = new double3[2];
		for (int i = 0; i < 8; i++)
			_childs[i] = nullptr;
		_pos[0] = min;
		_pos[1] = size;
		_dist = 0.0;
		_depth = depth;
	}
	~PRINode() {}
public:
	void clear() {
		for (int i = 0; i < 8; i++)
			if (_childs[i])
				_childs[i] = nullptr;
		delete[]_childs;
		delete[]_pos;
		_childs = nullptr;
		_pos = nullptr;
		free(this);
	}
	inline double3 center(void) { return _pos[0]; }
	inline double3 size(void) { return _pos[1]; }
	inline double3 setCenter(double3 p) { _pos[0] = p; }
	inline double3 setSize(double3 p) { _pos[1] = p; }
	inline bool isInside(double3 p) {
		return p.x >= _pos[0].x - _pos[1].x && p.x <= _pos[0].x + _pos[1].x &&
			p.y >= _pos[0].y - _pos[1].y && p.y <= _pos[0].y + _pos[1].y &&
			p.z >= _pos[0].z - _pos[1].z && p.z <= _pos[0].z + _pos[1].z;
	}
	inline double3 getCorner(int i) {
		switch (i) {
		case 0:
			return _pos[0];
		case 1:
			return make_double3(_pos[0].x + _pos[1].x, _pos[0].y, _pos[0].z);
		case 2:
			return make_double3(_pos[0].x, _pos[0].y + _pos[1].y, _pos[0].z);
		case 3:
			return make_double3(_pos[0].x + _pos[1].x, _pos[0].y + _pos[1].y, _pos[0].z);
		case 4:
			return make_double3(_pos[0].x, _pos[0].y, _pos[0].z + _pos[1].z);
		case 5:
			return make_double3(_pos[0].x + _pos[1].x, _pos[0].y, _pos[0].z + _pos[1].z);
		case 6:
			return make_double3(_pos[0].x, _pos[0].y + _pos[1].y, _pos[0].z + _pos[1].z);
		case 7:
			return make_double3(_pos[0].x + _pos[1].x, _pos[0].y + _pos[1].y, _pos[0].z + _pos[1].z);
		case 8:
			return _pos[0];
		case 9:
			return make_double3(_pos[0].x + _pos[1].x * 0.5, _pos[0].y, _pos[0].z);
		case 10:
			return make_double3(_pos[0].x, _pos[0].y + _pos[1].y * 0.5, _pos[0].z);
		case 11:
			return make_double3(_pos[0].x + _pos[1].x * 0.5, _pos[0].y + _pos[1].y * 0.5, _pos[0].z);
		case 12:
			return make_double3(_pos[0].x, _pos[0].y, _pos[0].z + _pos[1].z * 0.5);
		case 13:
			return make_double3(_pos[0].x + _pos[1].x * 0.5, _pos[0].y + _pos[1].y * 0.5, _pos[0].z + _pos[1].z * 0.5);
		case 14:
			return make_double3(_pos[0].x, _pos[0].y + _pos[1].y * 0.5, _pos[0].z + _pos[1].z * 0.5);
		case 15:
			return make_double3(_pos[0].x + _pos[1].x * 0.5, _pos[0].y + _pos[1].y * 0.5, _pos[0].z + _pos[1].z * 0.5);
		}
	}
	void subdivide(int num, double dist, double3 max) {
		double3 center = getCorner(num);
		if (center.x > max.x || center.y > max.y || center.z > max.z)
			return;
		_childs[num] = new PRINode(center, _pos[1] * 0.5, _depth + 1);
		_childs[num]->_dist = dist;
	}
	void draw(void) {
		double3 min = _pos[0] - _pos[1];
		double3 max = _pos[0] + _pos[1];
		glPushMatrix();
		glColor3f(1.0f, 1.0f, 1.0f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBegin(GL_LINES);
		glVertex3d(min.x, min.y, min.z);
		glVertex3d(min.x, min.y, max.z);
		glVertex3d(min.x, max.y, min.z);
		glVertex3d(min.x, max.y, max.z);
		glVertex3d(max.x, min.y, min.z);
		glVertex3d(max.x, min.y, max.z);
		glVertex3d(max.x, max.y, min.z);
		glVertex3d(max.x, max.y, max.z);
		glEnd();
		glTranslated(0, 0, min.z);
		glRectd(min.x, min.y, max.x, max.y);
		glTranslated(0, 0, max.z - min.z);
		glRectd(min.x, min.y, max.x, max.y);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glColor3f(1.0f, 1.0f, 1.0f);
		glPopMatrix();
	}
};

class SDF_PRITree
{
public:
	KDTree* _kdTree;
	SDFKernel* _kernel;
public:
	PRINode* _root;
	uint _maxDepth;
	double _error = 0.0000625;
public:
	SDF_PRITree();
	SDF_PRITree(KDTree* kdTree);
	~SDF_PRITree();
public:
	void clearUp(PRINode* node);
	void clear(void);
	void drawUp(PRINode* node);
	void draw(void);
	void drawBoundary(void);
	void save(int width, int height);
	PRINode* getNearestNode(double3 p);
	double getDistance(double3 p);
	double getDistWithGrad(double3 p, double3& gradient);
	bool comparison(PRINode* node, double* cdists);
	void buildTree(void);
	bool comparisonKernel(PRINode* node, double* cdists);
	void buildTreeKernel(void);
};
#endif
