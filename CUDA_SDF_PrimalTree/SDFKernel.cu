#include "DeviceFunc.cuh"

__device__ void warpReduce(volatile double* __restrict__ sdata, uint tid) {
	if (fabs(sdata[tid]) > fabs(sdata[tid + 32]))
		sdata[tid] = sdata[tid + 32];
	if (fabs(sdata[tid]) > fabs(sdata[tid + 16]))
		sdata[tid] = sdata[tid + 16];
	if (fabs(sdata[tid]) > fabs(sdata[tid + 8]))
		sdata[tid] = sdata[tid + 8];
	if (fabs(sdata[tid]) > fabs(sdata[tid + 4]))
		sdata[tid] = sdata[tid + 4];
	if (fabs(sdata[tid]) > fabs(sdata[tid + 2]))
		sdata[tid] = sdata[tid + 2];
	if (fabs(sdata[tid]) > fabs(sdata[tid + 1]))
		sdata[tid] = sdata[tid + 1];
}
__global__ void getDistToPointKernel(MeshDevice mesh, double3 p, double* __restrict__ output) {
	__shared__ double dists[BLOCKSIZE];
	uint tx = threadIdx.x;
	uint i = __umul24(BLOCKSIZE, __umul24(blockIdx.x, 2u)) + threadIdx.x;
	if (i >= mesh._numFace) {
		dists[tx] = 100.0;
		return;
	}

	if (i == 0)
		*output = 100.0;

	dists[tx] = getDist(p, mesh, i);
	if (i + BLOCKSIZE < mesh._numFace) {
		double _dist = getDist(p, mesh, i + BLOCKSIZE);
		if (fabs(dists[tx]) > fabs(_dist))
			dists[tx] = _dist;
	}
	__syncthreads();
#if BLOCKSIZE > 128
	for (uint s = BLOCKSIZE >> 1; s > 32; s >>= 1) {
		if (tx < s)
			if (fabs(dists[tx]) > fabs(dists[tx + s]))
				dists[tx] = dists[tx + s];
		__syncthreads();
}
#else
	if (tx < 64)
		if (fabs(dists[tx]) > fabs(dists[tx + 64]))
			dists[tx] = dists[tx + 64];
	__syncthreads();
#endif // BLOCKSIZE > 64

	if (tx < 32) {
		warpReduce(dists, tx);
		if (tx == 0)
			atomicMin_double(output, dists[0]);
	}
}
__global__ void getDistToPointsKernel(MeshDevice mesh, double3* points, double* __restrict__ output, uint pBlockSize) {
	__shared__ double dists[BLOCKSIZE];
	uint tx = threadIdx.x;
	uint index = blockIdx.x / pBlockSize;
	uint i = __umul24(BLOCKSIZE, __umul24(blockIdx.x % pBlockSize, 2u)) + threadIdx.x;
	if (i >= mesh._numFace) {
		dists[tx] = 100.0;
		return;
	}
	if (i == 0u)
		*(output + index) = 100.0;
	double3 p = *(points + index);
	dists[tx] = getDist(p, mesh, i);
	if (i + BLOCKSIZE < mesh._numFace) {
		double _dist = getDist(p, mesh, i + BLOCKSIZE);
		if (fabs(dists[tx]) > fabs(_dist))
			dists[tx] = _dist;
	}
	__syncthreads();
#if BLOCKSIZE > 128
	for (uint s = BLOCKSIZE >> 1; s > 32; s >>= 1) {
		if (tx < s)
			if (fabs(dists[tx]) > fabs(dists[tx + s]))
				dists[tx] = dists[tx + s];
		__syncthreads();
	}
#else
	if (tx < 64)
		if (fabs(dists[tx]) > fabs(dists[tx + 64]))
			dists[tx] = dists[tx + 64];
	__syncthreads();
#endif // BLOCKSIZE > 64

	if (tx < 32) {
		warpReduce(dists, tx);
		if (tx == 0)
			atomicMin_double(output + index, dists[0]);
	}
}
__global__ void saveDistance(MeshDevice mesh, double* __restrict__ output, uint res, uint pBlockSize) {
	__shared__ double dists[BLOCKSIZE];
	uint i = __umul24(BLOCKSIZE, __umul24(blockIdx.x % pBlockSize, 2u)) + threadIdx.x;
	if (i >= mesh._numFace) {
		dists[threadIdx.x] = 100.0;
		return;
	}
	if (i == 0)
		*(output + blockIdx.x / pBlockSize + blockIdx.y * res + blockIdx.z * res * res) = 100.0;
	double3 p = make_double3((double)(blockIdx.x / pBlockSize) * (1 / (double)res), (double)blockIdx.y * (1 / (double)res), (double)blockIdx.z * (1 / (double)res));

	dists[threadIdx.x] = getDist(p, mesh, i);
	if (i + BLOCKSIZE < mesh._numFace) {
		double _dist = getDist(p, mesh, i + BLOCKSIZE);
		if (fabs(dists[threadIdx.x]) > fabs(_dist))
			dists[threadIdx.x] = _dist;
	}
	__syncthreads();
#if BLOCKSIZE > 128
	for (uint s = BLOCKSIZE >> 1; s > 32; s >>= 1) {
		if (tx < s)
			if (fabs(dists[tx]) > fabs(dists[tx + s]))
				dists[tx] = dists[tx + s];
		__syncthreads();
	}
#else
	if (threadIdx.x < 64u)
		if (fabs(dists[threadIdx.x]) > fabs(dists[threadIdx.x + 64]))
			dists[threadIdx.x] = dists[threadIdx.x + 64];
	__syncthreads();
#endif // BLOCKSIZE > 64

	if (threadIdx.x < 32u) {
		warpReduce(dists, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicMin_double(output + blockIdx.x / pBlockSize + blockIdx.y * res + blockIdx.z * res * res, dists[0]);
	}
}

void SDFKernel::copyGPU(Mesh* mesh){
	VertexDevice* vertices;
	VDots* vds;
	double3* normals[7];

	_mesh._numFace = mesh->_faces.size();
	cudaMallocHost((void**)&vertices, _mesh._numFace * sizeof(VertexDevice));
	cudaMallocHost((void**)&vds, _mesh._numFace * sizeof(VDots));
	cudaMalloc((void**)&_mesh._vertices, _mesh._numFace * sizeof(VertexDevice));
	cudaMalloc((void**)&_mesh._vdots, _mesh._numFace * sizeof(VDots));
	for (int i = 0; i < 7; i++) {
		cudaMallocHost((void**)&normals[i], _mesh._numFace * sizeof(double3));
		cudaMalloc((void**)&_mesh._normals[i], _mesh._numFace * sizeof(double3));
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
void SDFKernel::Destroy(void) {
	if (!_mesh._numFace)
		return;
	cudaFree(_mesh._vertices);
	cudaFree(_mesh._vdots);
	for (int i = 0; i < 7; i++)
		cudaFree(_mesh._normals[i]);
	_mesh._numFace = 0;
}
double SDFKernel::getDistToPoint(double3 point) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	uint pblockNum = divup(_mesh._numFace, BLOCKSIZE * 2);
	double h_dist = DBL_MAX;
	double* d_dist;
	cudaMalloc((void**)&d_dist, sizeof(double));
	cudaMemcpy(d_dist, &h_dist, sizeof(double), cudaMemcpyHostToDevice);
	getDistToPointKernel << < divup(_mesh._numFace, BLOCKSIZE * 2), BLOCKSIZE >> > (_mesh, point, d_dist);
	cudaMemcpy(&h_dist, d_dist, sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_dist);
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("getDistToPoint(Chrono): %lf msec\n", (CNOW - timer) / 10000.0);
#endif
	return h_dist;
}
void SDFKernel::getDistToPoint(double3 point, double* output) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	uint pblockNum = divup(_mesh._numFace, BLOCKSIZE * 2);
	getDistToPointKernel << < divup(_mesh._numFace, BLOCKSIZE * 2), BLOCKSIZE >> > (_mesh, point, output);
	cudaDeviceSynchronize();
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("getDistToPoint: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
void SDFKernel::getDistToPoints(vector<double3> points, double* output) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	uint pblockNum = divup(_mesh._numFace, BLOCKSIZE * 2);
	double3* d_ps;
	double* d_dists;
	CUDA_CHECK(cudaMalloc((void**)&d_ps, points.size() * sizeof(double3)));
	CUDA_CHECK(cudaMalloc((void**)&d_dists, points.size() * sizeof(double)));
	CUDA_CHECK(cudaMemcpy(d_ps, &points[0], points.size() * sizeof(double3), cudaMemcpyHostToDevice));
	getDistToPointsKernel << < pblockNum * points.size(), BLOCKSIZE >> > (_mesh, d_ps, d_dists, pblockNum);
	CUDA_CHECK(cudaMemcpy(output, d_dists, sizeof(double) * points.size(), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(d_dists));
	CUDA_CHECK(cudaFree(d_ps));
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("getDistToPointsStream: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
void SDFKernel::getDistToPoints(double3* points, uint numPoint, double* output) {
	uint pblockNum = divup(_mesh._numFace, BLOCKSIZE * 2);
	getDistToPointsKernel << < pblockNum * numPoint, BLOCKSIZE, 0, 0 >> > (_mesh, points, output, pblockNum);
}
void SDFKernel::getDistRegion(uint size, double* output) {
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	uint pblockNum = divup(_mesh._numFace, BLOCKSIZE * 2);
	saveDistance << < dim3(pblockNum * size, size, size), BLOCKSIZE >> > (_mesh, output, size, pblockNum);
	cudaDeviceSynchronize();
#ifdef TESTTIMER
	cudaDeviceSynchronize();
	printf("saveDistance: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}