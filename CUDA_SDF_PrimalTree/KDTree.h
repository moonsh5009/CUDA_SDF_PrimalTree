#ifndef __KD_TREE_H__
#define __KD_TREE_H__

#pragma once
#include "Mesh.h"

class KDNode {
public:
	vector<Face*> _polygons;
	KDNode** _childs;
	double3* _pos;
	double _divPos;
	uint _divAxis;
	int _depth;
public:
	KDNode() {
		_pos = new double3[2];
		_childs = new KDNode* [2];
		for (int i = 0; i < 2; i++) {
			_pos[i] = make_double3(0.0);
			_childs[i] = nullptr;
		}
		_divPos = 0.0;
		_divAxis = 3;
		_depth = -1;
	}
	KDNode(double3 min, double3 max, int depth) {
		_pos = new double3[2];
		_childs = new KDNode * [2];
		for (int i = 0; i < 2; i++)
			_childs[i] = nullptr;
		_pos[0] = min;
		_pos[1] = max;
		_divPos = 0.0;
		_divAxis = 3;
		_depth = depth;
	}
	~KDNode(){}
public:
	inline double3 min(void) { return _pos[0]; }
	inline double3 max(void) { return _pos[1]; }
	inline void setMin(double3 p) { _pos[0] = p; }
	inline void setMax(double3 p) { _pos[1] = p; }
	void clear(void)
	{
		_polygons.clear();
		for (int i = 0; i < 2; i++)
			if (_childs[i] != nullptr)
				delete _childs[i];
		delete[] _pos;
		delete[] _childs;
		_pos = nullptr;
		_childs = nullptr;
	}
	void draw(void) {
		glPushMatrix();
		glLineWidth(3.0f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBegin(GL_LINES);
		glVertex3d(min().x, min().y, min().z);
		glVertex3d(min().x, min().y, max().z);
		glVertex3d(min().x, max().y, min().z);
		glVertex3d(min().x, max().y, max().z);
		glVertex3d(max().x, min().y, min().z);
		glVertex3d(max().x, min().y, max().z);
		glVertex3d(max().x, max().y, min().z);
		glVertex3d(max().x, max().y, max().z);
		glEnd();
		glTranslated(0, 0, min().z);
		glRectd(min().x, min().y, max().x, max().y);
		glTranslated(0, 0, max().z - min().z);
		glRectd(min().x, min().y, max().x, max().y);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glPopMatrix();

		glEnable(GL_FLAT);
		glColor3f(0.0f, 1.0f, 0.0f);
		glBegin(GL_TRIANGLES);
		for (auto f : _polygons)
			for (auto v : f->_vertices)
				glVertex3d(v->_pos.x, v->_pos.y, v->_pos.z);
		glEnd();
		glColor3f(1.0f, 1.0f, 1.0f);
		glLineWidth(1.0f);
		glDisable(GL_FLAT);
	}
};

class KDTree {
public:
	Mesh* _mesh;
	KDNode* _root;
	uint _maxDepth;
	uint _numNode;
public:
	vector<KDNode*>_testNodes;
public:
	KDTree();
	KDTree(Mesh* m);
	~KDTree();
public:
	void clearUp(KDNode* node);
	void clear(void);
	void drawUp(KDNode* node);
	void draw(void);
	void drawBoundary(void);
	void setMesh(Mesh* m);
	KDNode* getNearestNode(double3 p);
	void getDistToPoints(vector<double3> points, double* output);
	double getDistToPoint(double3 p);
	double getDistToPoint(double3 p, Face* f);
	bool isIntersectNode(KDNode* node, double3 p, double range);
	void subdivide(KDNode* node);
	void copyPolygons(void);
	void buildTree(void);
	void buildTree(double3 min, double3 max);
};

#endif
