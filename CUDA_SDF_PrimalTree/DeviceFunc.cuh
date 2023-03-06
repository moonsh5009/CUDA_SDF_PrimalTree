#ifndef __DEVICE_FUNCTION_H__
#define __DEVICE_FUNCTION_H__

#pragma once
#include "device_launch_parameters.h"
#include "SDFKernel.h"
#include "KDTreeKernel.h"

//#define CUNDEBUG

inline __device__ double atomicMin_double(double* address, double val)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (fabs(val) < fabs(__longlong_as_double(ret)))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long*)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}
inline __device__ double getDist(double3 p, MeshDevice mesh, uint i) {
	VertexDevice v = *(mesh._vertices + i);
	double v0pDotv01 = (p - v.p[0]) * (v.p[1] - v.p[0]);
	double v0pDotv02 = (p - v.p[0]) * (v.p[2] - v.p[0]);
	double v01Dotv01 = (mesh._vdots + i)->v01Dotv01;
	double v02Dotv02 = (mesh._vdots + i)->v02Dotv02;
	double v01Dotv02 = (mesh._vdots + i)->v01Dotv02;
	double v1pDotv12 = v0pDotv02 - v0pDotv01 - v01Dotv02 + v01Dotv01;
	double result = 0.0;
	bool term0 = v0pDotv01 <= 0;
	bool term1 = v01Dotv01 - v0pDotv01 <= 0;
	bool term2 = v0pDotv01 - v0pDotv02 - v01Dotv02 + v02Dotv02 <= 0;

	if (term0 && v0pDotv02 <= 0) {
		p -= v.p[0];
		v.p[0] = mesh._normals[0][i];
	}
	else if (v1pDotv12 <= 0 && term1) {
		p -= v.p[1];
		v.p[0] = mesh._normals[1][i];
	}
	else if (v02Dotv02 - v0pDotv02 <= 0 && term2) {
		p -= v.p[2];
		v.p[0] = mesh._normals[2][i];
	}
	else if (v0pDotv01 * v01Dotv02 - v0pDotv02 * v01Dotv01 >= 0 && !term0 && !term1) {
		p -= v.p[0];
		result -= v0pDotv01 * (v0pDotv01 / v01Dotv01);
		v.p[0] = mesh._normals[3][i];
	}
	else if ((v0pDotv01 - v01Dotv01) * (v02Dotv02 - v01Dotv02) - (v0pDotv02 - v01Dotv02) * (v01Dotv02 - v01Dotv01) >= 0 && !term2) {
		p -= v.p[1];
		result -= v1pDotv12 * v1pDotv12 / (v01Dotv01 + v02Dotv02 - v01Dotv02 - v01Dotv02);
		v.p[0] = mesh._normals[4][i];
	}
	else if (v0pDotv02 * v01Dotv02 - v0pDotv01 * v02Dotv02 >= 0) {
		p -= v.p[0];
		result -= v0pDotv02 * (v0pDotv02 / v02Dotv02);
		v.p[0] = mesh._normals[5][i];
	}
	else {
		result = mesh._normals[6][i] * (p - v.p[0]);
		return  fabs(result) * result;
	}
	result += p * p;
	if (v.p[0] * p < 0)
		return -result;
	return result;
}

#endif