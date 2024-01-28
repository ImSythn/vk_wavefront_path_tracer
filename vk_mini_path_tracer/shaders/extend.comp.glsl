#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require

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
//layout(binding = 1, set = 0, scalar) buffer rayBuffer
//{
//    float intersect[800 * 600];
//    int primitiveID[800 * 600];
//};
layout(binding = 4, set = 0) uniform accelerationStructureEXT tlas;

void main()
{
    const uvec2 resolution = uvec2(800, 600);
    const uvec2 pixel = gl_GlobalInvocationID.xy;

    // Get the index of this invocation in the buffer:
    uint indexOffset = resolution.x * pixel.y + pixel.x;
    const vec3 rayOrigin = rayData[indexOffset].origin;
    const vec3 rayDirection = rayData[indexOffset].direction;

    // Trace the ray and see if and where it intersects the scene!
    // First, initialize a ray query object:
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery,             // Ray query
        tlas,                                   // Top-level acceleration structure
        gl_RayFlagsOpaqueEXT,                   // Ray flags, here saying "treat all geometry as opaque"
        0xFF,                                   // 8-bit instance mask, here saying "trace against all instances"
        rayOrigin,                              // Ray origin
        0.0,                                    // Minimum t-value
        rayDirection,                           // Ray direction
        10000.0);                               // Maximum t-value

    // Start traversal, and loop over all ray-scene intersections. When this finishes,
    // rayQuery stores a "committed" intersection, the closest intersection (if any).
    while (rayQueryProceedEXT(rayQuery))
    {
    }

    // Get the t-value of the intersection (if there's no intersection, this will
    // be tMax = 10000.0). "true" says "get the committed intersection."
    rayData[indexOffset].intersect = rayQueryGetIntersectionTEXT(rayQuery, true);
    // If the ray intersects with a triangle give set its primitiveID
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
    {
        rayData[indexOffset].primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    }
    else
    {
        rayData[indexOffset].primitiveID = -1;
    }
}