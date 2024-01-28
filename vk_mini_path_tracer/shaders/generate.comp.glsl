#version 460
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

struct Ray
{
    vec3 origin;
    vec3 direction;
    float intersect;
    int primitiveID;
};

layout(binding = 0, set = 0, scalar) buffer rayBuffer
{
    Ray rayData[];
};

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
    // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
    rngState = rngState * 747796405 + 1;
    uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
    word = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

void main()
{
    const uvec2 resolution = uvec2(800, 600);
    const uvec2 pixel = gl_GlobalInvocationID.xy;

    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
    {
        return;
    }

    // State of the random number generator.
    uint rngState = resolution.x * pixel.y + pixel.x;  // Initial seed

    // This scene uses a right-handed coordinate system like the OBJ file format, where the
    // +x axis points right, the +y axis points up, and the -z axis points into the screen.
    // The camera is located at (-0.001, 1, 6).
    const vec3 cameraOrigin = vec3(-0.001, 1.0, 6.0);
    // Define the field of view by the vertical slope of the topmost rays:
    const float fovVerticalSlope = 1.0 / 5.0;

    // Rays always originate at the camera for now. In the future, they'll
    // bounce around the scene.
    vec3 rayOrigin = cameraOrigin;
    // Compute the direction of the ray for this pixel. To do this, we first
    // transform the screen coordinates to look like this, where a is the
    // aspect ratio (width/height) of the screen:
    //           1
    //    .------+------.
    //    |      |      |
    // -a + ---- 0 ---- + a
    //    |      |      |
    //    '------+------'
    //          -1
    const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    const vec2 screenUV = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
        -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis
    // Create a ray direction:
    vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection = normalize(rayDirection);

    // Get the index of this invocation in the buffer:
    uint indexOffset = resolution.x * pixel.y + pixel.x;
    rayData[indexOffset].origin = rayOrigin;
    rayData[indexOffset].direction = rayDirection;
}
