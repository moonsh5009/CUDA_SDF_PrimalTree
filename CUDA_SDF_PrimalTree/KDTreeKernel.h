#ifndef __KDTREE_KERNEL_H__
#define __KDTREE_KERNEL_H__

#pragma once
#include "KDTree.h"
#include "SDFKernel.h"

struct KDNodeDevice{
    double _divPos;
    uint _divAxis : 2;
    uint _isLLeaf : 1;
    uint _isRLeaf : 1;
    uint _left : 30;
    uint _right : 30;
};
struct KDLeafDevice {
    uint _numPolygon;
    uint _index;
};
class KDTreeKernel
{
    KDNodeDevice* _nodes;
    KDLeafDevice* _leaves;
    MeshDevice _polygons;
    double* _mins;
    double* _maxs;
public:
    uint _numNode;
    uint _numLeaf;
public:
	KDTreeKernel() {}
	~KDTreeKernel() {}
public:
    void buildTree(KDTree* tree);
    void Destroy(void);
    double getDistToPoint(double3 p);
    void getDistToPoints(vector<double3> points, double* output);
    void getDistToPoints(double3* points, uint numPoint, double* output);
};

#endif