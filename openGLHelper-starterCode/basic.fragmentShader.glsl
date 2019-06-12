#version 150

in vec3 viewPosition;
in vec3 viewNormal;

out vec4 c;

uniform vec4 La = vec4(1.0); //light ambient
uniform vec4 Ld = vec4(1.0); //light diffuse
uniform vec4 Ls = vec4(1.0); //light specular
uniform vec3 viewLightDirection;

uniform vec4 ka = vec4(vec3(0.2), 1.0); //mesh ambient
uniform vec4 kd = vec4(vec3(0.5), 1.0); //mesh diffuse
uniform vec4 ks = vec4(vec3(0.3), 1.0); //mesh specular
uniform float alpha = 1.0; //shininess

void main()
{
  //camera is at (0,0,0) after mv transformation
  vec3 eyedir = normalize(vec3(0,0,0) - viewPosition);

  //reflected light direction
  vec3 reflectDir = -reflect(viewLightDirection, viewNormal);

  //phong lighting
  float d = max(dot(viewLightDirection, viewNormal), 0.0f);
  float s = max(dot(reflectDir, eyedir), 0.0f);

  //final color
  c = vec4(vec3(ka * La + d * kd * Ld + pow(s, alpha) * ks * Ls), 1.0);
}

