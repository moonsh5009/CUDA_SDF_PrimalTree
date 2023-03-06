#include "Mesh.h"

bool Vertex::hasNbVertex(Vertex* v)
{
	for (auto nv : _nbVertices)
		if (nv == v)
			return true;
	return false;
}
int	Face::getIndex(Vertex* v)
{
	for (int i = 0; i < 3; i++)
		if (_vertices[i] == v)
			return i;
	return -1;
}
double3 Face::getEdgeNormal(int i)
{
	for (auto f : _vertices[i]->_nbFaces)
		if (this != f && f->getIndex(_vertices[(i + 1) % 3]) != -1)
			return getNormVec(f->_normal + _normal);
	return _normal;
}
void Mesh::loadObj(char* filename)
{
	FILE* fp;
	char header[256] = { 0 };
	double3 pos;
	int v_index[3];
	int index = 0;

	_minBoundary = make_double3(100000.0);
	_maxBoundary = make_double3(-100000.0);

	fopen_s(&fp, filename, "r");
	while (fscanf(fp, "%s %lf %lf %lf", header, &pos.x, &pos.y, &pos.z) != EOF)
		if (header[0] == 'v' && header[1] == NULL) {
			_minBoundary = minVec(_minBoundary, pos);
			_maxBoundary = maxVec(_maxBoundary, pos);
			_vertices.push_back(new Vertex(index++, pos));
		}
	printf("num. vertices : %d\n", _vertices.size());

	index = 0;
	fseek(fp, 0, SEEK_SET);
	while (fscanf(fp, "%s %d %d %d", header, &v_index[0], &v_index[1], &v_index[2]) != EOF)
		if (header[0] == 'f' && header[1] == NULL) {
			auto v0 = _vertices[v_index[0] - 1];
			auto v1 = _vertices[v_index[1] - 1];
			auto v2 = _vertices[v_index[2] - 1];
			_faces.push_back(new Face(index++, v0, v1, v2));
		}
	fclose(fp);
	printf("num. faces : %d\n", _faces.size());
	moveToCenter(_size);
	buildAdjacency();
	computeNormal();
}
void Mesh::buildAdjacency(void)
{
	for (auto v : _vertices) {
		v->_nbFaces.clear();
		v->_nbVertices.clear();
		v->_normal = make_double3(0.0);
	}
	for (auto f : _faces)
		for (int j = 0; j < 3; j++)
			f->_vertices[j]->_nbFaces.push_back(f);

	for (auto v : _vertices)
		for (auto nf : v->_nbFaces) {
			auto pivot = nf->getIndex(v);
			int other_id0 = (pivot + 1) % 3;
			int other_id1 = (pivot + 2) % 3;
			if (!v->hasNbVertex(nf->_vertices[other_id0]))
				v->_nbVertices.push_back(nf->_vertices[other_id0]);
			if (!v->hasNbVertex(nf->_vertices[other_id1]))
				v->_nbVertices.push_back(nf->_vertices[other_id1]);
		}
	printf("build adjacency list\n");
}
void Mesh::moveToCenter(double scale)
{
	double max_length = fmax(fmax(_maxBoundary.x - _minBoundary.x, _maxBoundary.y - _minBoundary.y), _maxBoundary.z - _minBoundary.z);
	auto center = (_maxBoundary + _minBoundary) / 2.0;
	double3 new_center = make_double3(0.0); //make_double3(0.5);

	for (auto v : _vertices) {
		auto pos = v->_pos;
		auto grad = pos - center;
		grad /= max_length;
		grad *= scale;
		pos = new_center;
		pos += grad;
		v->_pos = pos;
		_minBoundary = minVec(_minBoundary, pos);
		_maxBoundary = maxVec(_maxBoundary, pos);
	}
	printf("move to center\n");
}
void Mesh::computeNormal(void)
{
	for (auto f : _faces) {
		auto a = f->_vertices[0]->_pos;
		auto b = f->_vertices[1]->_pos;
		auto c = f->_vertices[2]->_pos;
		auto normal = Cross(a - b, a - c);
		Normalize(normal);
		f->_normal = normal;
		f->_vertices[0]->_normal += normal * AngleBetweenVectors(a - b, a - c);
		f->_vertices[1]->_normal += normal * AngleBetweenVectors(b - a, b - c);
		f->_vertices[2]->_normal += normal * AngleBetweenVectors(c - a, c - b);
	}

	for (auto v : _vertices)
		Normalize(v->_normal);
	printf("compute normal\n");
}
double Mesh::getDistToPoint(double3 p) {
#ifdef TESTTIMER
	//clock_t timer = clock();
	ctimer timer = CNOW;
	int computeNum = 0;
	printf("Compute Mesh Distance...");
#endif
	double minDist = DBL_MAX;
	for (auto f : _faces) {
#ifdef TESTTIMER
		computeNum++;
#endif
		double dist = getDistToPoint(p, f);
		if (fabs(minDist) > fabs(dist))
			minDist = dist;
	}
#ifdef TESTTIMER
	//printf(" Done!!\nElapsed TIme: %lf\n", (double)(clock() - timer));
	printf(" Done!!\nElapsed TIme: %lf msec\n", (CNOW - timer) / 10000.0);
	
	printf(" Compute Num: %d\n", computeNum);
#endif
	return minDist;
}
double Mesh::getDistToPoint(double3 p, Face* f) {
	double v0pDotv01 = (p - f->_vertices[0]->_pos) * (f->_vertices[1]->_pos - f->_vertices[0]->_pos);
	double v0pDotv02 = (p - f->_vertices[0]->_pos) * (f->_vertices[2]->_pos - f->_vertices[0]->_pos);
	double v01Dotv01 = LengthSquared(f->_vertices[1]->_pos - f->_vertices[0]->_pos);
	double v02Dotv02 = LengthSquared(f->_vertices[2]->_pos - f->_vertices[0]->_pos);
	double v01Dotv02 = (f->_vertices[1]->_pos - f->_vertices[0]->_pos) * (f->_vertices[2]->_pos - f->_vertices[0]->_pos);
	double v1pDotv12 = v0pDotv02 - v0pDotv01 - v01Dotv02 + v01Dotv01;
	double result = 0.0;
	uchar term0 = v0pDotv01 <= 0;
	uchar term1 = v01Dotv01 - v0pDotv01 <= 0;
	uchar term2 = v0pDotv01 - v0pDotv02 - v01Dotv02 + v02Dotv02 <= 0;
	double3 N;
	if (term0 && v0pDotv02 <= 0) {
		p -= f->_vertices[0]->_pos;
		N = f->_vertices[0]->_normal;
	}
	else if (v1pDotv12 <= 0 && term1) {
		p -= f->_vertices[1]->_pos;
		N = f->_vertices[1]->_normal;
	}
	else if (v02Dotv02 - v0pDotv02 <= 0 && term2) {
		p -= f->_vertices[2]->_pos;
		N = f->_vertices[2]->_normal;
	}
	else if (v0pDotv01 * v01Dotv02 - v0pDotv02 * v01Dotv01 >= 0 && !term0 && !term1) {
		p -= f->_vertices[0]->_pos;
		result -= v0pDotv01 * (v0pDotv01 / v01Dotv01);
		N = f->getEdgeNormal(0);
	}
	else if ((v0pDotv01 - v01Dotv01) * (v02Dotv02 - v01Dotv02) - (v0pDotv02 - v01Dotv02) * (v01Dotv02 - v01Dotv01) >= 0 && !term2) {
		p -= f->_vertices[1]->_pos;
		result -= v1pDotv12 * v1pDotv12 / (v01Dotv01 + v02Dotv02 - v01Dotv02 - v01Dotv02);
		N = f->getEdgeNormal(1);
	}
	else if (v0pDotv02 * v01Dotv02 - v0pDotv01 * v02Dotv02 >= 0) {
		p -= f->_vertices[0]->_pos;
		result -= v0pDotv02 * (v0pDotv02 / v02Dotv02);
		N = f->getEdgeNormal(2);
	}
	else {
		result = f->_normal * (p - f->_vertices[0]->_pos);
		return  fabs(result) * result;
	}
	result += p * p;
	if (N * p < 0)
		return -result;
	return result;
}
void Mesh::drawMesh(GLenum e) {
	glPushMatrix();

#ifdef TESTVIEWER
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif

	glEnable(GL_LIGHTING);
	glEnable(e);
	glColor3f(1.0f, 1.0f, 1.0f);
	for (auto f : _faces) {
		glBegin(GL_POLYGON);
		if (e == GL_SMOOTH) {
			for (auto v : f->_vertices) {
				glNormal3d(v->_normal.x, v->_normal.y, v->_normal.z);
				glVertex3d(v->_pos.x, v->_pos.y, v->_pos.z);
			}
		}
		else {
			glNormal3d(f->_normal.x, f->_normal.y, f->_normal.z);
			for (auto v : f->_vertices)
				glVertex3d(v->_pos.x, v->_pos.y, v->_pos.z);
		}
		glEnd();
	}
	glDisable(e);
	glDisable(GL_LIGHTING);

#ifdef TESTVIEWER
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif

	glPopMatrix();
}