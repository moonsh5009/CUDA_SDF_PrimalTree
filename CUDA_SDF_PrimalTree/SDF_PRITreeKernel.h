#ifndef __SDF_PRIMAL_TREE_KERNEL_H__
#define __SDF_PRIMAL_TREE_KERNEL_H__

#pragma once
#include "SDFKernel.h"
#include "SDF_PRITree.h"

class SDF_PRITreeKernel
{
public:
	KDTree* _kdTree;
	SDFKernel* _kernel;
	PRINode* _root;
	uint _maxDepth;
	double _error = 0.0000625;
public:
	SDF_PRITreeKernel();
	SDF_PRITreeKernel(KDTree* kdTree);
	~SDF_PRITreeKernel();
public:
	void save(int width, int height);
	PRINode* getNearestNode(double3 p);
	double getDistance(double3 p);
	double getDistWithGrad(double3 p, double3& gradient);
	bool comparison(PRINode* node, double* cdists);
	void getDistToPointsKernel(vector<double3>& h_points, double3* d_points, vector<double>& h_dists, double* d_dists);
	void buildTree(void);
};

#endif
