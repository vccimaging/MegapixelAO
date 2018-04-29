#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord_recon;
layout (location = 2) in vec2 aTexCoord_ref;

out vec2 TexCoord_recon;
out vec2 TexCoord_ref;

void main()
{
	gl_Position = vec4(aPos, 1.0);
	TexCoord_recon = aTexCoord_recon;
	TexCoord_ref = aTexCoord_ref;
}
