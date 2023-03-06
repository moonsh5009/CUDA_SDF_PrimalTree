#ifndef __MESH_H__
#define __MESH_H__

#pragma once
#include "Vec3.h"

class Vertex;
class Face;

class Vertex
{
public:
	bool			_flag;
	int				_index;
	double3			_pos;
	double3			_normal;
	vector<Face*>	_nbFaces;
	vector<Vertex*>	_nbVertices;
public:
	Vertex() { _flag = false; }
	Vertex(int index, double3 pos)
	{
		_flag = false;
		_index = index;
		_pos = pos;
	}
	~Vertex() {}
public:
	bool	hasNbVertex(Vertex* v);
};
class Face
{
public:
	int				_index;
	double3			_normal;
	vector<Vertex*>	_vertices;
public:
	Face() {}
	Face(int index, Vertex* v0, Vertex* v1, Vertex* v2)
	{
		_index = index;
		_vertices.push_back(v0);
		_vertices.push_back(v1);
		_vertices.push_back(v2);
	}
	~Face() {}
public:
	int		getIndex(Vertex* v);
	double3	getEdgeNormal(int i);
};

class Mesh
{
public:
	//double3			_pos;
	double3			_minBoundary;
	double3			_maxBoundary;
	double			_size;
	vector<Vertex*>	_vertices;
	vector<Face*>	_faces;
public:
	Mesh() {}
	Mesh(char* filename, double size = 0.6)
	{
		_size = size;
		loadObj(filename);
	}
	~Mesh() {}
public:
	void	loadObj(char* filename);
	void	moveToCenter(double scale);
	void	computeNormal(void);
	void	buildAdjacency(void);
	double	getDistToPoint(double3 p);
	double	getDistToPoint(double3 p, Face* f);
	void	drawMesh(GLenum e = GL_FLAT);
};

#endif
