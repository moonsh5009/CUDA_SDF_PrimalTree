#ifndef __VIEWER_H__
#define __VIEWER_H__

#pragma once
#include "KDTree.h"
#include "KDTreeKernel.h"

class Viewer
{
public:
	static Viewer* _ptr;
public:
	Mesh* _mesh;
	KDTree* _kdTree;
	KDTreeKernel* d_kdTree;
public:
	SDFKernel* _kernel;
	double3 _chkBall;
public:
	float _zoom;
	double2 _rotate;
	double2 _translate;
	double _deltaTime = 0.8;
public:
	int2 _last;
	uchar _btnStates[3];
public:
	Viewer() {}
	~Viewer() {}
public:
	static void GL_display(void);
	static void GL_reshape(int w, int h);
	static void GL_idle(void);
	static void GL_motion(int x, int y);
	static void GL_mouse(int button, int state, int x, int y);
	static void GL_keyboard(uchar key, int x, int y);
	static void GL_Skeyboard(int key, int x, int y);
public:
	void init(Viewer* ptr);
	void draw(void);
	void display(void);
	void reshape(int w, int h);
	void idle(void);
	void motion(int x, int y);
	void mouse(int button, int state, int x, int y);
	void keyboard(uchar key, int x, int y);
	void keyboard(int key, int x, int y);
	void show(int* argc, char** argv);
};

#endif