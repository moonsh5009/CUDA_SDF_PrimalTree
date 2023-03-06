#ifndef __SDF_KERNEL_H__
#define __SDF_KERNEL_H__

#pragma once
#include "Mesh.h"

struct VDots {
	double v01Dotv01;
	double v02Dotv02;
	double v01Dotv02;
};
struct VertexDevice {
	double3 p[3];
};
struct MeshDevice {
	VertexDevice* _vertices;
	VDots* _vdots;
	double3* _normals[7];
	uint _numFace;
};

class SDFKernel
{
	MeshDevice _mesh;
public:
	SDFKernel() {
		_mesh._numFace = 0;
	}
	SDFKernel(Mesh* mesh) {
		VertexDevice* vertices;
		VDots* vds;
		double3* normals[7];
		_mesh._numFace = mesh->_faces.size();
		cudaMalloc((void**)&_mesh._vertices, _mesh._numFace * sizeof(VertexDevice));
		cudaMalloc((void**)&_mesh._vdots, _mesh._numFace * sizeof(VDots));
		cudaMallocHost((void**)&vertices, _mesh._numFace * sizeof(VertexDevice));
		cudaMallocHost((void**)&vds, _mesh._numFace * sizeof(VDots));
		for (int i = 0; i < 7; i++) {
			cudaMalloc((void**)&_mesh._normals[i], _mesh._numFace * sizeof(double3));
			cudaMallocHost((void**)&normals[i], _mesh._numFace * sizeof(double3));
		}

		for (int i = 0; i < _mesh._numFace; i++) {
			vertices[i].p[0] = mesh->_faces[i]->_vertices[0]->_pos;
			vertices[i].p[1] = mesh->_faces[i]->_vertices[1]->_pos;
			vertices[i].p[2] = mesh->_faces[i]->_vertices[2]->_pos;
			vds[i].v01Dotv01 = LengthSquared(mesh->_faces[i]->_vertices[0]->_pos - mesh->_faces[i]->_vertices[1]->_pos);
			vds[i].v02Dotv02 = LengthSquared((mesh->_faces[i]->_vertices[0]->_pos - mesh->_faces[i]->_vertices[2]->_pos));
			vds[i].v01Dotv02 = (mesh->_faces[i]->_vertices[1]->_pos - mesh->_faces[i]->_vertices[0]->_pos) * (mesh->_faces[i]->_vertices[2]->_pos - mesh->_faces[i]->_vertices[0]->_pos);
			normals[0][i] = mesh->_faces[i]->_vertices[0]->_normal;
			normals[1][i] = mesh->_faces[i]->_vertices[1]->_normal;
			normals[2][i] = mesh->_faces[i]->_vertices[2]->_normal;
			normals[3][i] = mesh->_faces[i]->getEdgeNormal(0);
			normals[4][i] = mesh->_faces[i]->getEdgeNormal(1);
			normals[5][i] = mesh->_faces[i]->getEdgeNormal(2);
			normals[6][i] = mesh->_faces[i]->_normal;
		}

		cudaMemcpy(_mesh._vertices, vertices, sizeof(VertexDevice) * _mesh._numFace, cudaMemcpyHostToDevice);
		cudaMemcpy(_mesh._vdots, vds, sizeof(VDots) * _mesh._numFace, cudaMemcpyHostToDevice);
		for (uint i = 0; i < 7; i++)
			cudaMemcpy(_mesh._normals[i], normals[i], sizeof(double3) * _mesh._numFace, cudaMemcpyHostToDevice);

		cudaFreeHost(vds);
		cudaFreeHost(vertices);
		for (uint i = 0; i < 7; i++)
			cudaFreeHost(normals[i]);
	}
	~SDFKernel() {
		Destroy();
	}
public:
	void copyGPU(Mesh* mesh);
	void Destroy(void);
	double getDistToPoint(double3 point);
	void getDistToPoint(double3 point, double* output);
	void getDistToPoints(vector<double3> points, double* output);
	void getDistToPoints(double3* points, uint numPoint, double* output);
	void getDistRegion(uint size, double* output);
};


#endif