#include "Viewer.h"

Viewer* Viewer::_ptr = nullptr;

void Viewer::GL_display(void) {
	_ptr->display();
}
void Viewer::GL_reshape(int w, int h) {
	_ptr->reshape(w, h);
}
void Viewer::GL_idle(void) {
	_ptr->idle();
}
void Viewer::GL_motion(int x, int y) {
	_ptr->motion(x, y);
}
void Viewer::GL_mouse(int button, int state, int x, int y) {
	_ptr->mouse(button, state, x, y);
}
void Viewer::GL_keyboard(uchar key, int x, int y) {
	_ptr->keyboard(key, x, y);
}
void Viewer::GL_Skeyboard(int key, int x, int y) {
	_ptr->keyboard(key, x, y);
}
void Viewer::init(Viewer* ptr) {
	_zoom = 10.0;
	_rotate = make_double2(45, 0.0);
	_translate = make_double2(0.0, 0.0);
	_last = make_int2(0, 0);
	_btnStates[0] = 0;
	_btnStates[1] = 0;
	_btnStates[2] = 0;
	_ptr = ptr;

	_mesh = new Mesh("../../include/obj/bunny2.obj", 5.0);
	_kdTree = new KDTree(_mesh);
	//_kdTree->buildTree();
	_kdTree->buildTree(make_double3(-1.0), make_double3(1.0));

	d_kdTree = new KDTreeKernel();
	d_kdTree->buildTree(_kdTree);

	_kernel = new SDFKernel(_mesh);

	_chkBall = make_double3(0.0);
	_chkBall.z = -2.5;
}
void Viewer::draw(void) {
	glEnable(GL_LIGHT0);
	_mesh->drawMesh();
#ifdef TESTVIEWER
	_kdTree->drawBoundary();
	for (int i = 0; i < _kdTree->_testNodes.size(); i++) {
		if (i == 0)
			glColor3f(1.0f, 0.0f, 0.0f);
		else
			glColor3f(0.0f, 0.0f, 1.0f);
		_kdTree->_testNodes[i]->draw();
	}
	glPushMatrix();
	glTranslated(_chkBall.x, _chkBall.y, _chkBall.z);
	glColor3f(0.0f, 1.0f, 1.0f);
	glutSolidSphere(0.1, 10, 10);
	glPopMatrix();
#endif
}
void Viewer::display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();

	glTranslated(0.0, 0.0, -_zoom);
	glTranslated(_translate.x, _translate.y, 0.0);

	glRotated(_rotate.x, 1.0, 0.0, 0.0);
	glRotated(_rotate.y, 0.0, 1.0, 0.0);

	draw();
	glutSwapBuffers();
}

void Viewer::reshape(int w, int h) {
	if (w == 0)
		h = 1;
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double)w / (double)h, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glutPostRedisplay();
}

void Viewer::idle(void) {
	glutPostRedisplay();
}

void Viewer::motion(int x, int y) {
	int diff_x = x - _last.x;
	int diff_y = y - _last.y;
	_last.x = x;
	_last.y = y;
	if (_btnStates[2]) {
		_zoom -= 0.005 * diff_x;
	}
	else if (_btnStates[0]) {
		_rotate.x += 0.5 * diff_y;
		_rotate.y += 0.5 * diff_x;
	}
	else if (_btnStates[1]) {
		_translate.x += 0.05 * diff_x;
		_translate.y -= 0.05 * diff_y;
	}
	glutPostRedisplay();
}

void Viewer::mouse(int button, int state, int x, int y) {
	_last.x = x;
	_last.y = y;
	switch (button) {
	case GLUT_LEFT_BUTTON:
		_btnStates[0] = (GLUT_DOWN == state);
		break;
	case GLUT_MIDDLE_BUTTON:
		_btnStates[1] = (GLUT_DOWN == state);
		break;
	case GLUT_RIGHT_BUTTON:
		_btnStates[2] = (GLUT_DOWN == state);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Viewer::keyboard(uchar key, int x, int y) {
	switch (key) {
	case 'A':
	case 'a':
		_chkBall.x -= 0.05;
		break;
	case 'D':
	case 'd':
		_chkBall.x += 0.05;
		break;
	case 'S':
	case 's':
		_chkBall.z += 0.05;
		break;
	case 'W':
	case 'w':
		_chkBall.z -= 0.05;
		break;
	case 'Q':
	case 'q':
		exit(0);
		break;
	case 32:
		printf("------------------------------------\n");
		printf("Distance: %lf\n", _kdTree->getDistToPoint(_chkBall));
		printf("Distance: %lf\n", _mesh->getDistToPoint(_chkBall));
		printf("Distance: %lf\n", _kernel->getDistToPoint(_chkBall));
		printf("Distance: %lf\n", d_kdTree->getDistToPoint(_chkBall));
		printf("------------------------------------\n");
		break;
	default:
		break;
	}
	glutPostRedisplay();
}
void Viewer::keyboard(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_UP:
		_chkBall.y += 0.05;
		break;
	case GLUT_KEY_DOWN:
		_chkBall.y -= 0.05;
		break;
	case GLUT_KEY_LEFT:
		break;
	case GLUT_KEY_RIGHT:
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Viewer::show(int* argc, char** argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Camera Nevigation");
	glutDisplayFunc(_ptr->GL_display);
	glutReshapeFunc(_ptr->GL_reshape);
	glutIdleFunc(_ptr->GL_idle);
	glutMotionFunc(_ptr->GL_motion);
	glutMouseFunc(_ptr->GL_mouse);
	glutKeyboardFunc(_ptr->GL_keyboard);
	glutSpecialFunc(_ptr->GL_Skeyboard);
	glEnable(GL_DEPTH_TEST);
	glutMainLoop();
}