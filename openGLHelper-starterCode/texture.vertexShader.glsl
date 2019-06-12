#version 150

in vec3 pos;
in vec2 texCoord;

out vec2 tc;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

void main()
{
	//Compute the transformed and projected vertex position
	gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0f);

	//pass-through the tex coordinate
	tc = texCoord;
}