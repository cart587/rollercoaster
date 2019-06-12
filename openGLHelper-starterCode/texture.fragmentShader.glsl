#version 150

in vec2 tc; 
out vec4 c; //output color
uniform sampler2D textureImage;

void main()
{
	//compute final fragment color by looking into texturemap

	c = texture(textureImage, tc);
	//c.xy = tc;
}