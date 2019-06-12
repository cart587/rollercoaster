/*
  CSCI 420 Computer Graphics, USC
  Assignment 1: Height Fields
  C++ starter code

  Student username: cart587
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "basicPipelineProgram.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "openGLHeader.h"
#include "glutHeader.h"
#include "time.h"

#include <iostream>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

#if defined(WIN32) || defined(_WIN32)
  #ifdef _DEBUG
    #pragma comment(lib, "glew32d.lib")
  #else
    #pragma comment(lib, "glew32.lib")
  #endif
#endif

#if defined(WIN32) || defined(_WIN32)
  char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
  char shaderBasePath[1024] = "../openGLHelper-starterCode";
#endif

using namespace std;

int mousePos[2]; // x,y coordinate of the mouse position

int leftMouseButton = 0; // 1 if pressed, 0 if not 
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0; // 1 if pressed, 0 if not

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// state of the world
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };

int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 homework I";

ImageIO * heightmapImage;

//my variables
GLint h_modelViewMatrix, h_projectionMatrix;
OpenGLMatrix openGLMatrix;
BasicPipelineProgram pipelineProgram;
int makeAnimation = 0;
int imgCount = 0;
clock_t start;

//HW2 variables
GLuint splineVBO, splineVAO;
GLuint h_texture;
int camI = 0, camJ = 0;  
glm::vec3 splinePrevB, prevB;
vector<float> pos;
vector<float> cols;
vector<float> uvs;
vector<float> texPos;
GLuint texProgram;
GLint h_texModelViewMatrix, h_texProjectionMatrix;
BasicPipelineProgram texPipelineProgram;
GLuint texVBO, texVAO;

// represents one control point along the spline 
struct Point 
{
  double x;
  double y;
  double z;
};

// spline struct 
// contains how many control points the spline has, and an array of control points 
struct Spline 
{
  int numControlPoints;
  Point * points;
};

// the spline array 
Spline * splines;
// total number of splines 
int numSplines;

//Adds a triangle and the UV coordinates for texture mapping to the texture vbo
void addTriangle(float posA[3], float posB[3], float posC[3],
	float uvA[2], float uvB[2], float uvC[2])
{
	texPos.push_back(posA[0]); texPos.push_back(posA[1]); texPos.push_back(posA[2]);
	texPos.push_back(posB[0]); texPos.push_back(posB[1]); texPos.push_back(posB[2]);
	texPos.push_back(posC[0]); texPos.push_back(posC[1]); texPos.push_back(posC[2]);
	uvs.push_back(uvA[0]); uvs.push_back(uvA[1]);
	uvs.push_back(uvB[0]); uvs.push_back(uvB[1]);
	uvs.push_back(uvC[0]); uvs.push_back(uvC[1]);
}

//Adds a triangle and its normal to the vbo for the spline rail
void addTriangle(glm::vec3 posA, glm::vec3 posB, glm::vec3 posC,
	glm::vec3 col)
{
	pos.push_back(posA.x); pos.push_back(posA.y); pos.push_back(posA.z);
	pos.push_back(posB.x); pos.push_back(posB.y); pos.push_back(posB.z);
	pos.push_back(posC.x); pos.push_back(posC.y); pos.push_back(posC.z);
	cols.push_back(col.x); cols.push_back(col.y); cols.push_back(col.z); 
	cols.push_back(col.x); cols.push_back(col.y); cols.push_back(col.z);
	cols.push_back(col.x); cols.push_back(col.y); cols.push_back(col.z);
}

//Calculates the spline position using the Catmull Rom method using 4 control
//points and a u value.
Point catmullRomSpline(Point p0, Point p1, Point p2, Point p3, float u)
{
	//matrices stored column-wise
	glm::mat3x4 control = glm::mat3x4( p0.x, p1.x, p2.x, p3.x,
									   p0.y, p1.y, p2.y, p3.y,
									   p0.z, p1.z, p2.z, p3.z);

	glm::mat4 basis = glm::mat4 (-0.5, 1.0, -0.5, 0.0,
								 1.5, -2.5, 0.0, 1.0,
								 -1.5, 2.0, 0.5, 0.0,
								 0.5, -0.5, 0.0, 0.0 );

	glm::vec4 uVec = glm::vec4(pow(u,3), pow(u,2), u, 1.0);

	glm::vec3 spline = uVec * (basis * control);

	Point result = { spline.x, spline.y, spline.z };
	return result;
}

//Calculates the tangent at a specific spline position given the 4
//control points and a specific u.
Point tangentToSpline(Point p0, Point p1, Point p2, Point p3, float u)
{
	//matrices stored column-wise
	glm::mat3x4 control = glm::mat3x4(p0.x, p1.x, p2.x, p3.x,
									  p0.y, p1.y, p2.y, p3.y,
									  p0.z, p1.z, p2.z, p3.z);

	glm::mat4 basis = glm::mat4(-0.5, 1.0, -0.5, 0.0,
								1.5, -2.5, 0.0, 1.0,
								-1.5, 2.0, 0.5, 0.0,
								0.5, -0.5, 0.0, 0.0);

	glm::vec4 uVec = glm::vec4(3.0*pow(u, 2), 2.0*u, 1.0, 0.0);

	glm::vec3 tangent = uVec * (basis * control);

	Point result = { tangent.x, tangent.y, tangent.z };
	return result;
}

//HW2 starter code
int loadSplines(char * argv) 
{
  char * cName = (char *) malloc(128 * sizeof(char));
  FILE * fileList;
  FILE * fileSpline;
  int iType, i = 0, j, iLength;

  // load the track file 
  fileList = fopen(argv, "r");
  if (fileList == NULL) 
  {
    printf ("can't open file\n");
    exit(1);
  }
  
  // stores the number of splines in a global variable 
  fscanf(fileList, "%d", &numSplines);

  splines = (Spline*) malloc(numSplines * sizeof(Spline));

  // reads through the spline files 
  for (j = 0; j < numSplines; j++) 
  {
    i = 0;
    fscanf(fileList, "%s", cName);
    fileSpline = fopen(cName, "r");

    if (fileSpline == NULL) 
    {
      printf ("can't open file\n");
      exit(1);
    }

    // gets length for spline file
    fscanf(fileSpline, "%d %d", &iLength, &iType);

    // allocate memory for all the points
    splines[j].points = (Point *)malloc(iLength * sizeof(Point));
    splines[j].numControlPoints = iLength;

    // saves the data to the struct
    while (fscanf(fileSpline, "%lf %lf %lf", 
	   &splines[j].points[i].x, 
	   &splines[j].points[i].y, 
	   &splines[j].points[i].z) != EOF) 
    {
      i++;
    }
  }

  free(cName);

  return 0;
}

//HW2 starter code
int initTexture(const char * imageFilename, GLuint textureHandle)
{
  // read the texture image
  ImageIO img;
  ImageIO::fileFormatType imgFormat;
  ImageIO::errorType err = img.load(imageFilename, &imgFormat);

  if (err != ImageIO::OK) 
  {
    printf("Loading texture from %s failed.\n", imageFilename);
    return -1;
  }

  // check that the number of bytes is a multiple of 4
  if (img.getWidth() * img.getBytesPerPixel() % 4) 
  {
    printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
    return -1;
  }

  // allocate space for an array of pixels
  int width = img.getWidth();
  int height = img.getHeight();
  unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

  // fill the pixelsRGBA array with the image pixels
  memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
  for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++) 
    {
      // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
      pixelsRGBA[4 * (h * width + w) + 0] = 0; // red
      pixelsRGBA[4 * (h * width + w) + 1] = 0; // green
      pixelsRGBA[4 * (h * width + w) + 2] = 0; // blue
      pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

      // set the RGBA channels, based on the loaded image
      int numChannels = img.getBytesPerPixel();
      for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
        pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
    }

  // bind the texture
  glBindTexture(GL_TEXTURE_2D, textureHandle);

  // initialize the texture
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

  // generate the mipmaps for this texture
  glGenerateMipmap(GL_TEXTURE_2D);

  // set the texture parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // query support for anisotropic texture filtering
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
  printf("Max available anisotropic samples: %f\n", fLargest);
  // set anisotropic texture filtering
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

  // query for any errors
  GLenum errCode = glGetError();
  if (errCode != 0) 
  {
    printf("Texture initialization error. Error code: %d.\n", errCode);
    return -1;
  }
  
  // de-allocate the pixel array -- it is no longer needed
  delete [] pixelsRGBA;

  return 0;
}

//Adds all the triangles to form a rail between 2 positions on a spline
void addRailSegment(glm::vec3 sp, glm::vec3 sn, glm::vec3 sb, glm::vec3 ep, glm::vec3 en, glm::vec3 eb)
{
	float alpha = 0.5f;
	glm::vec3 v0 = sp + alpha * glm::normalize(-sn + sb);
	glm::vec3 v1 = sp + alpha * glm::normalize(sn + sb);
	glm::vec3 v2 = sp + alpha * glm::normalize(sn - sb);
	glm::vec3 v3 = sp + alpha * glm::normalize(-sn - sb);
	glm::vec3 v4 = ep + alpha * glm::normalize(-en + eb);
	glm::vec3 v5 = ep + alpha * glm::normalize(en + eb);
	glm::vec3 v6 = ep + alpha * glm::normalize(en - eb);
	glm::vec3 v7 = ep + alpha * glm::normalize(-en - eb);

	addTriangle(v0, v1, v4, sb);
	addTriangle(v1, v4, v5, sb);
	addTriangle(v1, v2, v5, sn);
	addTriangle(v2, v5, v6, sn);
	addTriangle(v2, v3, v6, -sb);
	addTriangle(v3, v6, v7, -sb);
	addTriangle(v0, v3, v4, -sn);
	addTriangle(v3, v4, v7, -sn);
}

//Creates all rail tubes from the beginning to the end of the spline curve.
//Creates and populates the vbo and vao representing spline rail.
void initSplinePoints(GLuint program) {
	
	int i,j;
	float step = 0.01;
	int NUM_SEGMENTS = 100;	// (1 / 0.001) = 1000 
	int numControlPoints = splines[0].numControlPoints - 4;

	for (i = 0; i < numControlPoints; i++) {
		Point p0 = splines[0].points[i];
		Point p1 = splines[0].points[i + 1];
		Point p2 = splines[0].points[i + 2];
		Point p3 = splines[0].points[i + 3];

		for (j = 0; j < NUM_SEGMENTS; j++) {
			float u = j * step;
			Point start = catmullRomSpline(p0, p1, p2, p3, u);
			Point temp = tangentToSpline(p0, p1, p2, p3, u);
			glm::vec3 startPoint = glm::vec3(start.x, start.y, start.z);
			glm::vec3 startTangent = glm::normalize(glm::vec3(temp.x, temp.y, temp.z));
			glm::vec3 startN, startB;

			Point end = catmullRomSpline(p0, p1, p2, p3, u + step);
			Point eTemp = tangentToSpline(p0, p1, p2, p3, u + step);
			glm::vec3 endPoint = glm::vec3(end.x, end.y, end.z);
			glm::vec3 endTangent = glm::normalize(glm::vec3(eTemp.x, eTemp.y, eTemp.z));
			glm::vec3 endN, endB;

			if (i == 0 && j == 0) {
				glm::vec3 v = glm::normalize(glm::vec3(1.0, 1.0, 1.0));
				startN = glm::normalize(glm::cross(startTangent, v));
				startB = glm::normalize(glm::cross(startTangent, startN));
				endN = glm::normalize(glm::cross(startB, endTangent));
				endB = glm::normalize(glm::cross(endTangent, endN));
				splinePrevB = startB;
			}
			else {
				startN = glm::normalize(glm::cross(splinePrevB, startTangent));
				startB = glm::normalize(glm::cross(startTangent, startN));
				endN = glm::normalize(glm::cross(startB, endTangent));
				endB = glm::normalize(glm::cross(endTangent, endN));
				splinePrevB = startB;
			}
			
			addRailSegment(startPoint, startN, startB, endPoint, endN, endB);
		}
	}

	//create vbo
	pipelineProgram.Bind();
	glGenBuffers(1, &splineVBO);
	glBindBuffer(GL_ARRAY_BUFFER, splineVBO);
	glBufferData(GL_ARRAY_BUFFER, (pos.size() + cols.size()) * sizeof(float), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, pos.size() * sizeof(float), pos.data());
	glBufferSubData(GL_ARRAY_BUFFER, pos.size() * sizeof(float), cols.size() * sizeof(float), cols.data());

	//create vao
	glGenVertexArrays(1, &splineVAO);
	glBindVertexArray(splineVAO);
	glBindBuffer(GL_ARRAY_BUFFER, splineVBO);

	//enable position shader
	GLuint h_position = glGetAttribLocation(program, "position");
	glEnableVertexAttribArray(h_position);
	const void * offset = (const void *)0;
	GLsizei stride = 0;
	GLboolean isNormalized = GL_FALSE;
	glVertexAttribPointer(h_position, 3, GL_FLOAT, isNormalized, stride, offset);

	//enable color/normal shader
	GLuint h_color = glGetAttribLocation(program, "normal");
	glEnableVertexAttribArray(h_color);
	offset = (const void *)(sizeof(float) * pos.size());
	stride = 0;
	isNormalized = GL_FALSE;
	glVertexAttribPointer(h_color, 3, GL_FLOAT, isNormalized, stride, offset);

	glBindVertexArray(0);
}

//Creates a grid of textures to be displayed as a ground plane for the roller coaster simulation
//Creates and populates the vbo and vao representing the texture
void initGroundPlane()
{
	glGenTextures(1, &h_texture);
	int code = initTexture("textures/lava2_square.jpg", h_texture);

	if (code != 0)
	{
		printf("Error loading the texture image.\n");
		exit(EXIT_FAILURE);
	}

	int plane_width = 60;
	int plane_height = 60;
	float corner = 300.0f;
	float texSize = 10.0f;
	float texTopLeft[2] = { 0.0f, 1.0f };
	float texTopRight[2] = { 1.0f, 1.0f };
	float texBottomLeft[2] = { 0.0f, 0.0f };
	float texBottomRight[2] = { 1.0f, 0.0f };


	for (int i = 0; i < plane_width; i++) {
		for (int j = 0; j < plane_height; j++)
		{
			float topLeft[3] = { -corner + texSize * i, corner - texSize * j, -10.0f };
			float topRight[3] = { -corner + texSize * (i + 1), corner - texSize * j , -10.0f };
			float bottomLeft[3] = { -corner + texSize * i, corner - texSize * (j + 1), -10.0f };
			float bottomRight[3] = { -corner + texSize * (i + 1), corner - texSize * (j + 1), -10.0f };

			addTriangle(topLeft, bottomLeft, bottomRight, texTopLeft, texBottomLeft, texBottomRight);
			addTriangle(topLeft, bottomRight, topRight, texTopLeft, texBottomRight, texTopRight);
		}
	}

	

	//create vbo
	texPipelineProgram.Bind();
	glGenBuffers(1, &texVBO);
	glBindBuffer(GL_ARRAY_BUFFER, texVBO);
	glBufferData(GL_ARRAY_BUFFER, (texPos.size() + uvs.size()) * sizeof(float), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, texPos.size() * sizeof(float), texPos.data());
	glBufferSubData(GL_ARRAY_BUFFER, texPos.size() * sizeof(float), uvs.size() * sizeof(float), uvs.data());

	//create vao
	glGenVertexArrays(1, &texVAO);
	glBindVertexArray(texVAO);
	glBindBuffer(GL_ARRAY_BUFFER, texVBO);

	//enable position shader variable
	GLuint h_position = glGetAttribLocation(texProgram, "pos");
	glEnableVertexAttribArray(h_position);
	const void * offset = (const void *)0;
	GLsizei stride = 0;
	GLboolean isNormalized = GL_FALSE;
	glVertexAttribPointer(h_position, 3, GL_FLOAT, isNormalized, stride, offset);

	// get location index of the texCoord shader variable
	GLuint loc = glGetAttribLocation(texProgram, "texCoord");
	glEnableVertexAttribArray(loc); // enable the texCoord attribute
	
	offset = (const void*) (sizeof(float) * texPos.size()); 
	stride = 0;
	glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, stride, offset);
}

//Draws the spline rail representing the roller coaster
void drawSpline()
{
	GLint first = 0;
	GLsizei count = pos.size()/3;//num of vertices

	pipelineProgram.Bind();
	glBindVertexArray(splineVAO);
	glDrawArrays(GL_TRIANGLES, first, count);
	glBindVertexArray(0);
}

//Draws the texture plane representing the ground
void drawTexture()
{
	GLint first = 0;
	GLsizei count = texPos.size() / 3;

	texPipelineProgram.Bind();
	glBindVertexArray(texVAO);
	glDrawArrays(GL_TRIANGLES, first, count);
}

// write a screenshot to the specified filename
void saveScreenshot(const char * filename)
{
  unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
}

//Normalizes a Point so that the length/magnitude of Point is 1
Point Normalize(Point p)
{
	double scalar = 1.0 / (sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2)));
	Point n = { scalar*p.x, scalar*p.y, scalar*p.z };
	return n;
}

//Constantly updates the camera position to simulate camera moving along the spline
void moveCameraAlongSpline()
{
	glm::vec3 n, b;
	int NUM_SEGMENTS = 1000;
	float step = .001f;

	if (camJ > NUM_SEGMENTS) {
		camI++;
		camJ = 0;
	}
		
	if (camI > splines[0].numControlPoints - 4)
		camI = 0;

	Point p0 = splines[0].points[camI];
	Point p1 = splines[0].points[camI + 1];
	Point p2 = splines[0].points[camI + 2];
	Point p3 = splines[0].points[camI + 3];
	float u = step * camJ;

	Point pos = catmullRomSpline(p0, p1, p2, p3, u);
	Point tangent = Normalize(tangentToSpline(p0, p1, p2, p3, u));
	glm::vec3 t = glm::vec3(tangent.x, tangent.y, tangent.z);

	if (camJ == 0 && camI == 0) {
		glm::vec3 v = glm::vec3( 1.0, 1.0, 1.0 );
		n = glm::normalize(glm::cross(t, v));
		b = glm::normalize(glm::cross(t, n));
	}
	else {
		n = glm::normalize(glm::cross(prevB, t));
		b = glm::normalize(glm::cross(t, n));
	}
	prevB = b;

	float ride = 1.0f;

	float posX = pos.x + ride * n.x;
	float posY = pos.y + ride * n.y;
	float posZ = pos.z + ride * n.z;

	openGLMatrix.LookAt(posX, posY, posZ, t.x + posX, t.y + posY, t.z + posZ, n.x, n.y, n.z);

	camJ++;
}

void setTextureUnit(GLint unit)
{
	glActiveTexture(unit); // select the active texture unit
	// get a handle to the “textureImage” shader variable
	GLint h_textureImage = glGetUniformLocation(texProgram, "textureImage");
	// deem the shader variable “textureImage” to read from texture unit “unit”
	glUniform1i(h_textureImage, unit - GL_TEXTURE0);
}

void displayFunc()
{
  // render some stuff...

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	openGLMatrix.SetMatrixMode(OpenGLMatrix::ModelView);
	openGLMatrix.LoadIdentity();

	//Position Camera
	moveCameraAlongSpline();

	//upload light direction vector to GPU
	float view[16];
	openGLMatrix.GetMatrix(view);

	GLint h_viewLightDirection = glGetUniformLocation(pipelineProgram.GetProgramHandle(), "viewLightDirection");
	float lightDirection[3] = { 0, 1, 0 }; //Sun at noon

	glm::mat4 glmView = glm::make_mat4(view);
	glm::vec4 glmLight = glm::vec4(lightDirection[0], lightDirection[1], lightDirection[2], 0.0f);
	glm::vec4 glmViewLight = glmView * glmLight;
	float viewLightDirection[3] = { glmViewLight.x, glmViewLight.y, glmViewLight.z };

	glUniform3fv(h_viewLightDirection, 1, viewLightDirection);

	//upload normal matrix
	GLint h_normalMatrix = glGetUniformLocation(pipelineProgram.GetProgramHandle(), "normalMatrix");
	float n[16];
	openGLMatrix.GetNormalMatrix(n);
	GLboolean isRowMajor = GL_FALSE;
	glUniformMatrix4fv(h_normalMatrix, 1, isRowMajor, n);

	//Translate, Scale, and Rotate object
	openGLMatrix.Translate(landTranslate[0], landTranslate[1], landTranslate[2]);
	openGLMatrix.Scale(landScale[0], landScale[1], landScale[2]);
	openGLMatrix.Rotate(landRotate[0], 1.0, 0.0, 0.0);
	openGLMatrix.Rotate(landRotate[1], 0.0, 1.0, 0.0);
	openGLMatrix.Rotate(landRotate[2], 0.0, 0.0, 1.0);

	//upload modelView matrix
	pipelineProgram.Bind(); //call once right before glUniformMatrix4fv
	float m[16];
	openGLMatrix.GetMatrix(m);
	glUniformMatrix4fv(h_modelViewMatrix, 1, GL_FALSE, m);
	
	//upload projection matrix
	openGLMatrix.SetMatrixMode(OpenGLMatrix::Projection);
	float p[16];
	openGLMatrix.GetMatrix(p);
	glUniformMatrix4fv(h_projectionMatrix, 1, GL_FALSE, p);
	
	//upload matrices for textures pipelines
	texPipelineProgram.Bind();
	glUniformMatrix4fv(h_texModelViewMatrix, 1, GL_FALSE, m);
	glUniformMatrix4fv(h_texProjectionMatrix, 1, GL_FALSE, p);

	// select the active texture unit
	setTextureUnit(GL_TEXTURE0); // it is safe to always use GL_TEXTURE0
	// select the texture to use (“h_texture” was generated by glGenTextures)
	glBindTexture(GL_TEXTURE_2D, h_texture);

	//Draw vao contents based on mode
	drawSpline();
	drawTexture();
	glutSwapBuffers();
}

void idleFunc()
{
  // do some stuff... 

  // for example, here, you can save the screenshots to disk (to make the animation)
	if (makeAnimation == 1) {

		//ensure screenshot taken every 1/15 secs
		if (clock() >= start + (CLOCKS_PER_SEC / 15)) {
			char *name = (char*)malloc(sizeof(char) * 7);
			sprintf(name, "%.3d.jpg", imgCount);

			saveScreenshot(name);
			start = clock();
			imgCount++;
		}
		
	}

	if (imgCount >= 1000) {
		makeAnimation = 0;
	}

  // make the screen update 
  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  // setup perspective matrix...
  openGLMatrix.SetMatrixMode(OpenGLMatrix::Projection);
  openGLMatrix.LoadIdentity();
  openGLMatrix.Perspective(60.0, (1.0f * w) / h, 0.01, 5000);
  openGLMatrix.SetMatrixMode(OpenGLMatrix::ModelView);
}

void mouseMotionDragFunc(int x, int y)
{
  // mouse has moved and one of the mouse buttons is pressed (dragging)

  // the change in mouse position since the last invocation of this function
  int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

  switch (controlState)
  {
    // translate the landscape
    case TRANSLATE:
      if (leftMouseButton)
      {
        // control x,y translation via the left mouse button
        landTranslate[0] += mousePosDelta[0] * 0.01f;
        landTranslate[1] -= mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z translation via the middle mouse button
        landTranslate[2] += mousePosDelta[1] * 0.01f;
      }
      break;

    // rotate the landscape
    case ROTATE:
      if (leftMouseButton)
      {
        // control x,y rotation via the left mouse button
        landRotate[0] += mousePosDelta[1];
        landRotate[1] += mousePosDelta[0];
      }
      if (middleMouseButton)
      {
        // control z rotation via the middle mouse button
        landRotate[2] += mousePosDelta[1];
      }
      break;

    // scale the landscape
    case SCALE:
      if (leftMouseButton)
      {
        // control x,y scaling via the left mouse button
        landScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
        landScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z scaling via the middle mouse button
        landScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
  // mouse has moved
  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y)
{
  // a mouse button has has been pressed or depressed

  // keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
      leftMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_MIDDLE_BUTTON:
      middleMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_RIGHT_BUTTON:
      rightMouseButton = (state == GLUT_DOWN);
    break;
  }

  // keep track of whether CTRL and SHIFT keys are pressed
  switch (glutGetModifiers())
  {
    case GLUT_ACTIVE_CTRL:
      controlState = TRANSLATE;
    break;

    case GLUT_ACTIVE_SHIFT:
      controlState = SCALE;
    break;

    // if CTRL and SHIFT are not pressed, we are in rotate mode
    default:
      controlState = ROTATE;
    break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

//Added a few keys to switch between the render modes
void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27: // ESC key
      exit(0); // exit the program
    break;

    case ' ':
      cout << "You pressed the spacebar." << endl;
    break;

    case 'x':
      // take a screenshot
      saveScreenshot("screenshot.jpg");
    break;

	case 'a':
		makeAnimation = 1;
		start = clock();
  }
}



void initScene(int argc, char *argv[])
{
	// load the splines from the provided filename
	loadSplines(argv[1]);

	printf("Loaded %d spline(s).\n", numSplines);
	for (int i = 0; i < numSplines; i++)
		printf("Num control points in spline %d: %d.\n", i, splines[i].numControlPoints);

  glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

  // do additional initialization here...
  glEnable(GL_DEPTH_TEST);

  //init pipeline
  pipelineProgram.Init("../openGLHelper-starterCode");
  pipelineProgram.Bind();
  GLuint program = pipelineProgram.GetProgramHandle();
  h_modelViewMatrix = glGetUniformLocation(program, "modelViewMatrix");
  h_projectionMatrix = glGetUniformLocation(program, "projectionMatrix");

  //init texture pipeline
  texPipelineProgram.InitTexture("../openGLHelper-starterCode");
  texPipelineProgram.Bind();
  texProgram = texPipelineProgram.GetProgramHandle();
  h_texModelViewMatrix = glGetUniformLocation(texProgram, "modelViewMatrix");
  h_texProjectionMatrix = glGetUniformLocation(texProgram, "projectionMatrix");

  //create VBOs and VAOs
  initGroundPlane();
  initSplinePoints(program);
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		printf("usage: %s <trackfile>\n", argv[0]);
		exit(0);
	}


  cout << "Initializing GLUT..." << endl;
  glutInit(&argc,argv);

  cout << "Initializing OpenGL..." << endl;

  #ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);  
  glutCreateWindow(windowTitle);

  cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  // tells glut to use a particular display function to redraw 
  glutDisplayFunc(displayFunc);
  // perform animation inside idleFunc
  glutIdleFunc(idleFunc);
  // callback for mouse drags
  glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  glutMouseFunc(mouseButtonFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

  // init glew
  #ifdef __APPLE__
    // nothing is needed on Apple
  #else
    // Windows, Linux
    GLint result = glewInit();
    if (result != GLEW_OK)
    {
      cout << "error: " << glewGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
    }
  #endif

  // do initialization
  initScene(argc, argv);

  // sink forever into the glut loop
  glutMainLoop();
}


