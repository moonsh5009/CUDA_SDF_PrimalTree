#ifndef __PHYS_VECTOR_DYNAMIC_H__
#define __PHYS_VECTOR_DYNAMIC_H__

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <Windows.h>
#include <vector>
#include <algorithm>
#include "../../include/GL/freeglut.h"
#include <chrono>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

//#define TESTTIMER
//#define TESTVIEWER

#define BLOCKSIZE	128
#define CNOW		system_clock::now()

#define CUDEBUG
#if defined(CUNDEBUG)
#define CUDA_CHECK(x)	(x)
#elif defined(CUDEBUG)
#define CUDA_CHECK(x)	do {\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (e != cudaSuccess) { \
			printf("cuda failure %s:%d: '%s'\n", \
				__FILE__, __LINE__, cudaGetErrorString(e)); \
			/*exit(1);*/ \
		}\
	} while(0)
#endif

using namespace std;
using namespace cv;
using namespace chrono;
typedef unsigned int uint;
typedef unsigned char uchar;
typedef system_clock::time_point ctimer;

inline double3 SCALAR_TO_COLOR(double val)
{
	// T fColorMap[3][3] = {{0.960784314,0.498039216,0.011764706},{0,0,0},{0,0.462745098,0.88627451}};
	double fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };	//Red->Blue
	//T fColorMap[4][3] = {{0.15,0.35,1.0},{1.0,1.0,1.0},{1.0,0.5,0.0},{0.0,0.0,0.0}};
	double v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	double t = v - low;
	double3 color;
	color.x = ((fColorMap[low][0]) * (1 - t) + (fColorMap[high][0]) * t);
	color.y = ((fColorMap[low][1]) * (1 - t) + (fColorMap[high][1]) * t);
	color.z = ((fColorMap[low][2]) * (1 - t) + (fColorMap[high][2]) * t);
	return color;
}
inline double ssqrt(double d) {
	if (d == 0.0)
		return d;
	return d / sqrt(fabs(d));
}
inline double Lerp(double d0, double d1, double x) {
	double result = d0 * (1.0 - x) + d1 * x;
	return result;
}
inline double Lerp(double d0, double d1) {
	double result = ((d0)+(d1)) * 0.5;
	return result;
}
inline double biLerp(double d0, double d1, double d2, double d3, double2 p) {
	double x1 = d0 * (1.0 - p.x) + d1 * p.x;
	double x2 = d2 * (1.0 - p.x) + d3 * p.x;
	double result = x1 * (1.0 - p.y) + x2 * p.y;
	return result;
}
inline double biLerp(double d0, double d1, double d2, double d3) {
	double result = (d0 + d1 + d2 + d3) * 0.25;
	return result;
}
inline double triLerp(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double3 p) {
	double x1 = d0 * (1.0 - p.x) + d1 * p.x;
	double x2 = d2 * (1.0 - p.x) + d3 * p.x;
	double x3 = d4 * (1.0 - p.x) + d5 * p.x;
	double x4 = d6 * (1.0 - p.x) + d7 * p.x;
	double y1 = x1 * (1.0 - p.y) + x2 * p.y;
	double y2 = x3 * (1.0 - p.y) + x4 * p.y;
	double result = y1 * (1.0 - p.z) + y2 * p.z;
	return result;
}
inline double triLerp(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7) {
	double result = (d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7) * 0.125;
	return result;
}

inline __host__ __device__ uint divup(uint x, uint y) {
	return (x + y - 1) / y;
}
inline __host__ __device__  double3 operator+(double3* a, const double3 b)
{
	return make_double3(a->x + b.x, a->y + b.y, a->z + b.z);
}
inline __host__ __device__  double3 operator-(double3* a, double3 b)
{
	return make_double3(a->x - b.x, a->y - b.y, a->z - b.z);
}
inline __host__ __device__ double operator*(double3* a, double3 b)
{
	return a->x * b.x + a->y * b.y + a->z * b.z;
}
inline __host__ __device__  double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__  double3 operator-(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ double operator*(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
	return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
	return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator*(double3 a, double b)
{
	return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator/(double3 a, double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ void operator-=(double3& a, double3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ void operator*=(double3& a, double3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ void operator/=(double3& a, double3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
inline __host__ __device__ void operator+=(double3& a, double b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}
inline __host__ __device__ void operator-=(double3& a, double b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}
inline __host__ __device__ void operator*=(double3& a, double b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}
inline __host__ __device__ void operator/=(double3& a, double b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
inline __host__ __device__ bool operator==(const double3 a, const double3 b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator!=(const double3 a, const double3 b)
{
	return a.x != b.x && a.y != b.y && a.z != b.z;
}
inline __host__ __device__ bool operator==(const double3 a, const double b)
{
	return a.x == b && a.y == b && a.z == b;
}
inline __host__ __device__ bool operator!=(const double3 a, const double b)
{
	return a.x != b && a.y != b && a.z != b;
}
inline __host__ __device__ bool operator<(const double3 a, const double3 b)
{
	return a.x < b.x && a.y < b.y && a.z < b.z;
}
inline __host__ __device__ bool operator>(const double3 a, const double3 b)
{
	return a.x > b.x && a.y > b.y && a.z > b.z;
}
inline __host__ __device__ bool operator<=(const double3 a, const double3 b)
{
	return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}
inline __host__ __device__ bool operator>=(const double3 a, const double3 b)
{
	return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}
inline __host__ __device__ bool operator<(const double3 a, const double b)
{
	return a.x < b && a.y < b && a.z < b;
}
inline __host__ __device__ bool operator>(const double3 a, const double b)
{
	return a.x > b && a.y > b && a.z > b;
}
inline __host__ __device__ bool operator<=(const double3 a, const double b)
{
	return a.x <= b&& a.y <= b&& a.z <= b;
}
inline __host__ __device__ bool operator>=(const double3 a, const double b)
{
	return a.x >= b && a.y >= b && a.z >= b;
}
inline __host__ __device__ double3 make_double3(double s)
{
	return make_double3(s, s, s);
}
inline __host__ __device__ double3 make_double3(double2 a)
{
	return make_double3(a.x, a.y, 0.0f);
}
inline __host__ __device__ double3 make_double3(double2 a, double s)
{
	return make_double3(a.x, a.y, s);
}
inline __host__ __device__ double3 make_double3(double4 a)
{
	return make_double3(a.x, a.y, a.z);
}
inline __host__ __device__ double3 minVec(double3 a, double3 b)
{
	return make_double3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ double3 maxVec(double3 a, double3 b)
{
	return make_double3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ void Print(double3 a)
{
	printf("%f, %f, %f\n", a.x, a.y, a.z);
}
inline __host__ __device__ double LengthSquared(double3 a)
{
	return a * a;
}
inline __host__ __device__ double LengthSquared(double x, double y, double z)
{
	return x * x + y * y + z * z;
}
inline __host__ __device__ double Length(double3 a)
{
	return sqrt(a * a);
}
inline __host__ __device__ void Normalize(double3& a)
{
	double norm = Length(a);
	if (norm != 0) {
		a.x = a.x / norm;
		a.y = a.y / norm;
		a.z = a.z / norm;
	}
}
inline __host__ __device__ double3 getNormVec(double3 a)
{
	double norm = Length(a);
	if (norm == 0)
		return a;
	return make_double3(a.x / norm, a.y / norm, a.z / norm);
}
inline __host__ __device__ double3 Cross(double3 a, double3 b)
{
	return make_double3(a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x);
}
inline __host__ __device__ double AngleBetweenVectors(double3 a, double3 b)
{
	double dot, cross;
	double3 tmp;
	dot = a * b;
	tmp = Cross(a, b);
	cross = Length(tmp);
	return atan2(cross, dot);
}

#endif