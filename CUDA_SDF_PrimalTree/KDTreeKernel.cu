#include "DeviceFunc.cuh"

__device__ bool isIntersectNode(double* p, double* min, double* max, double range)
{
	double minDist = 0.0;
	for (int i = 0; i < 3; i++) {
		double tmp = p[i] - min[i];
		if (tmp < 0)
			minDist += tmp * tmp;
		else if ((tmp = p[i] - max[i]) > 0)
			minDist += tmp * tmp;
	}
	return (minDist <= range);
}
__global__ void getDistToPointsKernel(double* points, uint numPoints, MeshDevice polygons, KDNodeDevice* nodes, KDLeafDevice* leaves, uint numLeaf, double* mins, double* maxs, double* output) {
	uint i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numPoints)
		return;
	uint curr = 0u;
	double p[3];
	p[0] = *(points + 3u * i);
	p[1] = *(points + 1u + 3u * i);
	p[2] = *(points + 2u + 3u * i);
	//Get Nearest Node
	while (1) {
		KDNodeDevice node = nodes[curr];
		if (p[node._divAxis] < node._divPos) {
			curr = node._left;
			if (node._isLLeaf)
				break;
		}
		else {
			curr = node._right;
			if (node._isRLeaf)
				break;
		}
	}
	KDLeafDevice leaf = leaves[curr];
	//Get Range
	double range = DBL_MAX;
	for (uint stride = 0u; stride < leaf._numPolygon; stride++) {
		VertexDevice vs = polygons._vertices[leaf._index + stride];
		for (uint n = 0u; n < 3u; n++) {
			double dist = LengthSquared((p[0] - vs.p[n].x)
				, (p[1] - vs.p[n].y), (p[2] - vs.p[n].z));

			if (range > dist)
				range = dist;
		}
	}
	//Get Distance
	double minDist = DBL_MAX;
	for (uint stride = 0u; stride < numLeaf; stride++) {
		leaf = leaves[stride];
		if (isIntersectNode(p, mins + 3 * stride, maxs + 3 * stride, range)) {
			for (uint pstride = 0u; pstride < leaf._numPolygon; pstride++) {
				double dist = getDist(make_double3(p[0], p[1], p[2]), polygons, leaf._index + pstride);
				if (fabs(minDist) > fabs(dist))
					minDist = dist;
			}
		}
	}
	output[i] = minDist;
}
void KDTreeKernel::buildTree(KDTree* tree)
{
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	vector<KDNodeDevice> h_nodes;
	vector<KDLeafDevice> h_leaves;
	vector<double> h_mins;
	vector<double> h_maxs;

	vector<VertexDevice> h_vertices;
	vector<VDots> h_vdots;
	vector<double3> h_normals[7];

	vector<KDNode*> queue;
	_numNode = 1;
	_numLeaf = 0;
	queue.push_back(tree->_root);
	while (queue.size()) {
		KDNode* node = queue[0];
		queue.erase(queue.begin());
		if (node->_childs[0]) {
			KDNodeDevice nodeDevice;
			nodeDevice._divAxis = node->_divAxis;
			nodeDevice._divPos = node->_divPos;
			if (node->_childs[0]->_childs[0]) {
				nodeDevice._isLLeaf = 0;
				nodeDevice._left = _numNode++;
			}
			else {
				nodeDevice._isLLeaf = 1;
				nodeDevice._left = _numLeaf++;
			}
			if (node->_childs[1]->_childs[0]) {
				nodeDevice._isRLeaf = 0;
				nodeDevice._right = _numNode++;
			}
			else {
				nodeDevice._isRLeaf = 1;
				nodeDevice._right = _numLeaf++;
			}
			h_nodes.push_back(nodeDevice);
			queue.push_back(node->_childs[0]);
			queue.push_back(node->_childs[1]);
		}
		else {
			KDLeafDevice nodeDevice;
			h_mins.push_back(node->min().x);
			h_mins.push_back(node->min().y);
			h_mins.push_back(node->min().z);
			h_maxs.push_back(node->max().x);
			h_maxs.push_back(node->max().y);
			h_maxs.push_back(node->max().z);
			nodeDevice._index = h_vertices.size();
			//Copy Faces
			for (auto f : node->_polygons) {
				VertexDevice v;
				VDots vds;
				double3 N[7];
				v.p[0] = f->_vertices[0]->_pos;
				v.p[1] = f->_vertices[1]->_pos;
				v.p[2] = f->_vertices[2]->_pos;
				vds.v01Dotv01 = LengthSquared(f->_vertices[0]->_pos - f->_vertices[1]->_pos);
				vds.v02Dotv02 = LengthSquared((f->_vertices[0]->_pos - f->_vertices[2]->_pos));
				vds.v01Dotv02 = (f->_vertices[1]->_pos - f->_vertices[0]->_pos) * (f->_vertices[2]->_pos - f->_vertices[0]->_pos);
				N[0] = f->_vertices[0]->_normal;
				N[1] = f->_vertices[1]->_normal;
				N[2] = f->_vertices[2]->_normal;
				N[3] = f->getEdgeNormal(0);
				N[4] = f->getEdgeNormal(1);
				N[5] = f->getEdgeNormal(2);
				N[6] = f->_normal;
				h_vertices.push_back(v);
				h_vdots.push_back(vds);
				for (int i = 0; i < 7; i++)
					h_normals[i].push_back(N[i]);
			}
			nodeDevice._numPolygon = h_vertices.size() - nodeDevice._index;
			h_leaves.push_back(nodeDevice);
		}
	}
	_polygons._numFace = h_vertices.size();

	cudaMalloc((void**)&_nodes, _numNode * sizeof(KDNodeDevice));
	cudaMalloc((void**)&_leaves, _numLeaf * sizeof(KDLeafDevice));
	CUDA_CHECK(cudaMalloc((void**)&_mins, h_mins.size() * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void**)&_maxs, h_maxs.size() * sizeof(double)));
	cudaMemcpy(_nodes, &h_nodes[0], _numNode * sizeof(KDNodeDevice), cudaMemcpyHostToDevice);
	cudaMemcpy(_leaves, &h_leaves[0], _numLeaf * sizeof(KDLeafDevice), cudaMemcpyHostToDevice);
	CUDA_CHECK(cudaMemcpy(_mins, &h_mins[0], h_mins.size() * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(_maxs, &h_maxs[0], h_maxs.size() * sizeof(double), cudaMemcpyHostToDevice));

	cudaMalloc((void**)&_polygons._vertices, _polygons._numFace * sizeof(VertexDevice));
	cudaMalloc((void**)&_polygons._vdots, _polygons._numFace * sizeof(VDots));
	cudaMemcpy(_polygons._vertices, &h_vertices[0], _polygons._numFace *sizeof(VertexDevice), cudaMemcpyHostToDevice);
	cudaMemcpy(_polygons._vdots, &h_vdots[0], _polygons._numFace * sizeof(VDots), cudaMemcpyHostToDevice);
	for (int i = 0; i < 7; i++) {
		cudaMalloc((void**)&_polygons._normals[i], _polygons._numFace * sizeof(double3));
		cudaMemcpy(_polygons._normals[i], &h_normals[i][0], _polygons._numFace * sizeof(double3), cudaMemcpyHostToDevice);
	}
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("GPU KDtree Build: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
void KDTreeKernel::Destroy(void) {
	cudaFree(_nodes);
	cudaFree(_leaves);
	cudaFree(_mins);
	cudaFree(_maxs);
	cudaFree(_polygons._vertices);
	cudaFree(_polygons._vdots);
	for (int i = 0; i < 7; i++)
		cudaFree(_polygons._normals[i]);
	_numNode = _numLeaf = _polygons._numFace = 0;
}
double KDTreeKernel::getDistToPoint(double3 p) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	double* d_point;
	double* d_dist;
	double h_dist;
	cudaMalloc((void**)&d_point, sizeof(double) * 3);
	cudaMalloc((void**)&d_dist, sizeof(double));
	cudaMemcpy(d_point, &p, sizeof(double) * 3, cudaMemcpyHostToDevice);
	getDistToPointsKernel << <1, 1 >> > (d_point, 1, _polygons, _nodes, _leaves, _numLeaf, _mins, _maxs, d_dist);
	cudaMemcpy(&h_dist, d_dist, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_point);
	cudaFree(d_dist);
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("GPU KDtree getDistToPoint: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
	return h_dist;
}
void KDTreeKernel::getDistToPoints(vector<double3> points, double* output) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	double* d_points, * d_dists;
	cudaMalloc((void**)&d_points, sizeof(double) * 3 * points.size());
	cudaMalloc((void**)&d_dists, sizeof(double) * points.size());
	cudaMemcpy(d_points, &points[0], sizeof(double) * 3 * points.size(), cudaMemcpyHostToDevice);
	getDistToPointsKernel << <divup(points.size(), BLOCKSIZE), BLOCKSIZE >> > (d_points, points.size(), _polygons, _nodes, _leaves, _numLeaf, _mins, _maxs, d_dists);
	cudaMemcpy(output, d_dists, sizeof(double) * points.size(), cudaMemcpyDeviceToHost);
	cudaFree(d_points);
	cudaFree(d_dists);
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("GPU KDtree getDistToPoints: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
void KDTreeKernel::getDistToPoints(double3* points, uint numPoint, double* output) {
	getDistToPointsKernel << <divup(numPoint, BLOCKSIZE), BLOCKSIZE, 0, 0 >> > ((double*)points, numPoint, _polygons, _nodes, _leaves, _numLeaf, _mins, _maxs, output);
}