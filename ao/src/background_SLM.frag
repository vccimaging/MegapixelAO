#version 330 core

out vec4 FragColor;

in vec2 TexCoord_recon;
in vec2 TexCoord_ref;

// texture sampler
uniform sampler2D tex_recon;
uniform sampler2D tex_ref;

void main()
{
    vec4 temp = texture(tex_recon, TexCoord_recon) + texture(tex_ref, TexCoord_ref);
    FragColor = temp;
    FragColor = mod(temp, 1.0f);
}
