#ifndef __SDF_PRIMAL_TREE_KDKERNEL_H__
#define __SDF_PRIMAL_TREE_KDKERNEL_H__

#pragma once
#include "KDTreeKernel.h"
#include "SDF_PRITree.h"

class SDF_PRITreeKDKernel
{
public:
	KDTree* _kdTree;
	KDTreeKernel* _kernel;
	PRINode* _root;
	uint _maxDepth;
	double _error = 0.0000625;
public:
	SDF_PRITreeKDKernel();
	SDF_PRITreeKDKernel(KDTree* kdTree);
	~SDF_PRITreeKDKernel();
public:
	void save(int width, int height);
	PRINode* getNearestNode(double3 p);
	double getDistance(double3 p);
	double getDistWithGrad(double3 p, double3& gradient);
	bool comparison(PRINode* node, double* cdists);
	void getDistToPointsKernel(vector<double3>& h_points, double3* d_points, double* h_dists, double* d_dists);
	void buildTree(void);
};

#endif
