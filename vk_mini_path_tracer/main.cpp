// Copyright 2020-2021 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#include <array>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <nvh/fileoperations.hpp>  // For nvh::loadFile
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>  // For nvvk::DescriptorSetContainer
#include <nvvk/error_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>        // For nvvk::RaytracingBuilderKHR
#include <nvvk/resourceallocator_vk.hpp>  // For NVVK memory allocators
#include <nvvk/shaders_vk.hpp>            // For nvvk::createShaderModule
#include <nvvk/structs_vk.hpp>            // For nvvk::make
#include "vma/vk_mem_alloc.h"             // For buffer allocation

static const uint64_t render_width = 800;
static const uint64_t render_height = 600;
static const uint32_t workgroup_width = 16;
static const uint32_t workgroup_height = 8;
static const uint32_t max_iterations = 64;


VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
	VkCommandBufferAllocateInfo cmdAllocInfo = nvvk::make<VkCommandBufferAllocateInfo>();
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandPool = cmdPool;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer cmdBuffer;
	NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
	VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
	return cmdBuffer;
}

void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
	NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
	VkSubmitInfo submitInfo = nvvk::make<VkSubmitInfo>();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;
	NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	NVVK_CHECK(vkQueueWaitIdle(queue));
	vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
	VkBufferDeviceAddressInfo addressInfo = nvvk::make<VkBufferDeviceAddressInfo>();
	addressInfo.buffer = buffer;
	return vkGetBufferDeviceAddress(device, &addressInfo);
}

// Function to create a compute pipeline
VkPipeline CreateComputePipeline(const nvvk::Context& context, VkShaderModule shaderModule, VkDescriptorSetLayout descriptorSetLayout)
{
	VkPipelineShaderStageCreateInfo shaderStageCreateInfo = nvvk::make<VkPipelineShaderStageCreateInfo>();
	shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStageCreateInfo.module = shaderModule;
	shaderStageCreateInfo.pName = "main";

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = nvvk::make<VkPipelineLayoutCreateInfo>();
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

	VkPipelineLayout pipelineLayout;
	NVVK_CHECK(vkCreatePipelineLayout(context, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

	VkComputePipelineCreateInfo pipelineCreateInfo = nvvk::make<VkComputePipelineCreateInfo>();
	pipelineCreateInfo.stage = shaderStageCreateInfo;
	pipelineCreateInfo.layout = pipelineLayout;

	VkPipeline pipeline;
	NVVK_CHECK(vkCreateComputePipelines(context, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline));

	return pipeline;
}

struct Ray
{
	glm::vec3 origin;
	glm::vec3 direction;
	float intersect;
	int primitiveID;
};

int main(int argc, const char** argv)
{
	// Create the Vulkan context, consisting of an instance, device, physical device, and queues.
	nvvk::ContextCreateInfo deviceInfo;  // One can modify this to load different extensions or pick the Vulkan core version
	deviceInfo.apiMajor = 1;             // Specify the version of Vulkan we'll use
	deviceInfo.apiMinor = 2;
	// Required by KHR_acceleration_structure; allows work to be offloaded onto background threads and parallelized
	deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = nvvk::make<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
	deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &asFeatures);
	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = nvvk::make<VkPhysicalDeviceRayQueryFeaturesKHR>();
	deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

	nvvk::Context context;     // Encapsulates device state in a single object
	context.init(deviceInfo);  // Initialize the context
	// Device must support acceleration structures and ray queries:
	assert(asFeatures.accelerationStructure == VK_TRUE && rayQueryFeatures.rayQuery == VK_TRUE);

	// Create the allocator
	nvvk::ResourceAllocatorDedicated allocator;
	allocator.init(context, context.m_physicalDevice);

	//// Create a buffer
	//VkDeviceSize       bufferSizeBytes  = render_width * render_height * 3 * sizeof(float);
	//VkBufferCreateInfo bufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
	//bufferCreateInfo.size               = bufferSizeBytes;
	//bufferCreateInfo.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	//// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that the CPU can read this buffer's memory.
	//// VK_MEMORY_PROPERTY_HOST_CACHED_BIT means that the CPU caches this memory.
	//// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means that the CPU side of cache management
	//// is handled automatically, with potentially slower reads/writes.
	//nvvk::Buffer buffer = allocator.createBuffer(bufferCreateInfo,                         //
	//                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT       //
	//                                                 | VK_MEMORY_PROPERTY_HOST_CACHED_BIT  //
	//                                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	// Create phase 1 buffer to store the rays
	VkDeviceSize generateBufferSizeBytes = render_width * render_height * sizeof(Ray); // origin and direction per pixel
	VkBufferCreateInfo generateBufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
	generateBufferCreateInfo.size = generateBufferSizeBytes;
	generateBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	nvvk::Buffer generateBuffer = allocator.createBuffer(generateBufferCreateInfo,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		| VK_MEMORY_PROPERTY_HOST_CACHED_BIT
		| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	// Create phase 2 buffer to store the rays
	VkDeviceSize extendBufferSizeBytes = render_width * render_height * sizeof(float) * sizeof(int); // origin and direction per pixel
	VkBufferCreateInfo extendBufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
	extendBufferCreateInfo.size = extendBufferSizeBytes;
	extendBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	nvvk::Buffer extendBuffer = allocator.createBuffer(extendBufferCreateInfo,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		| VK_MEMORY_PROPERTY_HOST_CACHED_BIT
		| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	// Create phase 3 buffer to store the rays
	VkDeviceSize shadeBufferSizeBytes = render_width * render_height * sizeof(float) * 3 * 2; // origin and direction per pixel
	VkBufferCreateInfo shadeBufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
	shadeBufferCreateInfo.size = shadeBufferSizeBytes;
	shadeBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	nvvk::Buffer shadeBuffer = allocator.createBuffer(shadeBufferCreateInfo,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		| VK_MEMORY_PROPERTY_HOST_CACHED_BIT
		| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	// Create phase 4 buffer to store the rays
	VkDeviceSize connectBufferSizeBytes = render_width * render_height * sizeof(float) * 3 * 2; // origin and direction per pixel
	VkBufferCreateInfo connectBufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
	connectBufferCreateInfo.size = connectBufferSizeBytes;
	connectBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	nvvk::Buffer connectBuffer = allocator.createBuffer(connectBufferCreateInfo,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		| VK_MEMORY_PROPERTY_HOST_CACHED_BIT
		| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	// Load the mesh of the first shape from an OBJ file
	const std::string        exePath(argv[0], std::string(argv[0]).find_last_of("/\\") + 1);
	std::vector<std::string> searchPaths = { exePath + PROJECT_RELDIRECTORY, exePath + PROJECT_RELDIRECTORY "..",
											exePath + PROJECT_RELDIRECTORY "../..", exePath + PROJECT_NAME };
	tinyobj::ObjReader       reader;  // Used to read an OBJ file
	reader.ParseFromFile(nvh::findFile("scenes/CornellBox-Original-Merged.obj", searchPaths));
	assert(reader.Valid());  // Make sure tinyobj was able to parse this file
	const std::vector<tinyobj::real_t>   objVertices = reader.GetAttrib().GetVertices();
	const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file
	assert(objShapes.size() == 1);                                          // Check that this file has only one shape
	const tinyobj::shape_t& objShape = objShapes[0];                        // Get the first shape
	// Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
	std::vector<uint32_t> objIndices;
	objIndices.reserve(objShape.mesh.indices.size());
	for (const tinyobj::index_t& index : objShape.mesh.indices)
	{
		objIndices.push_back(index.vertex_index);
	}

	// Create the command pool
	VkCommandPoolCreateInfo cmdPoolInfo = nvvk::make<VkCommandPoolCreateInfo>();
	cmdPoolInfo.queueFamilyIndex = context.m_queueGCT;
	VkCommandPool cmdPool;
	NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));

	// Upload the vertex and index buffers to the GPU.
	nvvk::Buffer vertexBuffer, indexBuffer;
	{
		// Start a command buffer for uploading the buffers
		VkCommandBuffer uploadCmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
		// We get these buffers' device addresses, and use them as storage buffers and build inputs.
		const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
		vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
		indexBuffer = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);

		EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
		allocator.finalizeAndReleaseStaging();
	}

	// Describe the bottom-level acceleration structure (BLAS)
	std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;
	{
		nvvk::RaytracingBuilderKHR::BlasInput blas;
		// Get the device addresses of the vertex and index buffers
		VkDeviceAddress vertexBufferAddress = GetBufferDeviceAddress(context, vertexBuffer.buffer);
		VkDeviceAddress indexBufferAddress = GetBufferDeviceAddress(context, indexBuffer.buffer);
		// Specify where the builder can find the vertices and indices for triangles, and their formats:
		VkAccelerationStructureGeometryTrianglesDataKHR triangles = nvvk::make<VkAccelerationStructureGeometryTrianglesDataKHR>();
		triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		triangles.vertexData.deviceAddress = vertexBufferAddress;
		triangles.vertexStride = 3 * sizeof(float);
		triangles.maxVertex = static_cast<uint32_t>(objVertices.size() / 3 - 1);
		triangles.indexType = VK_INDEX_TYPE_UINT32;
		triangles.indexData.deviceAddress = indexBufferAddress;
		triangles.transformData.deviceAddress = 0;  // No transform
		// Create a VkAccelerationStructureGeometryKHR object that says it handles opaque triangles and points to the above:
		VkAccelerationStructureGeometryKHR geometry = nvvk::make<VkAccelerationStructureGeometryKHR>();
		geometry.geometry.triangles = triangles;
		geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		blas.asGeometry.push_back(geometry);
		// Create offset info that allows us to say how many triangles and vertices to read
		VkAccelerationStructureBuildRangeInfoKHR offsetInfo;
		offsetInfo.firstVertex = 0;
		offsetInfo.primitiveCount = static_cast<uint32_t>(objIndices.size() / 3);  // Number of triangles
		offsetInfo.primitiveOffset = 0;
		offsetInfo.transformOffset = 0;
		blas.asBuildOffsetInfo.push_back(offsetInfo);
		blases.push_back(blas);
	}
	// Create the BLAS
	nvvk::RaytracingBuilderKHR raytracingBuilder;
	raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
	raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	// Create an instance pointing to this BLAS, and build it into a TLAS:
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	{
		VkAccelerationStructureInstanceKHR instance{};
		instance.accelerationStructureReference = raytracingBuilder.getBlasDeviceAddress(0);  // The address of the BLAS in `blases` that this instance points to
		// Set the instance transform to the identity matrix:
		instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
		instance.instanceCustomIndex = 0;  // 24 bits accessible to ray shaders via rayQueryGetIntersectionInstanceCustomIndexEXT
		// Used for a shader offset index, accessible via rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // How to trace this instance
		instance.mask = 0xFF;
		instances.push_back(instance);
	}
	raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	// Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl:
	// 0 - a storage buffer (the buffer `buffer`)
	// 1 - an acceleration structure (the TLAS)
	// 2 - a storage buffer (the vertex buffer)
	// 3 - a storage buffer (the index buffer)
	nvvk::DescriptorSetContainer descriptorSetContainer(context);
	descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(4, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	// Create a layout from the list of bindings
	descriptorSetContainer.initLayout();
	// Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
	descriptorSetContainer.initPool(1);
	// Create a simple pipeline layout from the descriptor set layout:
	descriptorSetContainer.initPipeLayout();

	//// Create a descriptor set layout for Phase 1
	//nvvk::DescriptorSetContainer generateDescriptorSetContainer(context);
	//generateDescriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	//generateDescriptorSetContainer.initLayout();
	//generateDescriptorSetContainer.initPool(1);
	//generateDescriptorSetContainer.initPipeLayout();

	// Write values into the descriptor set.
	std::array<VkWriteDescriptorSet, 7> writeDescriptorSets;
	// 0
	VkDescriptorBufferInfo generateDescriptorBufferInfo{};
	generateDescriptorBufferInfo.buffer = generateBuffer.buffer;    // The VkBuffer object
	generateDescriptorBufferInfo.range = generateBufferSizeBytes;  // The length of memory to bind; offset is 0.
	writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0 /*set index*/, 0 /*binding*/, &generateDescriptorBufferInfo);
	// 1
	VkDescriptorBufferInfo  extendDescriptorBufferInfo{};
	extendDescriptorBufferInfo.buffer = extendBuffer.buffer;    // The VkBuffer object
	extendDescriptorBufferInfo.range = extendBufferSizeBytes;  // The length of memory to bind; offset is 0.
	writeDescriptorSets[1] = descriptorSetContainer.makeWrite(0 /*set index*/, 1 /*binding*/, &extendDescriptorBufferInfo);
	// 2
	VkDescriptorBufferInfo shadeDescriptorBufferInfo{};
	shadeDescriptorBufferInfo.buffer = shadeBuffer.buffer;    // The VkBuffer object
	shadeDescriptorBufferInfo.range = shadeBufferSizeBytes;  // The length of memory to bind; offset is 0.
	writeDescriptorSets[2] = descriptorSetContainer.makeWrite(0 /*set index*/, 2 /*binding*/, &shadeDescriptorBufferInfo);
	// 3
	VkDescriptorBufferInfo connectDescriptorBufferInfo{};
	connectDescriptorBufferInfo.buffer = connectBuffer.buffer;    // The VkBuffer object
	connectDescriptorBufferInfo.range = connectBufferSizeBytes;  // The length of memory to bind; offset is 0.
	writeDescriptorSets[3] = descriptorSetContainer.makeWrite(0 /*set index*/, 3 /*binding*/, &connectDescriptorBufferInfo);
	// 4
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAS = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
	VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
	descriptorAS.accelerationStructureCount = 1;
	descriptorAS.pAccelerationStructures = &tlasCopy;
	writeDescriptorSets[4] = descriptorSetContainer.makeWrite(0, 4, &descriptorAS);
	// 5
	VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
	vertexDescriptorBufferInfo.buffer = vertexBuffer.buffer;
	vertexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	writeDescriptorSets[5] = descriptorSetContainer.makeWrite(0, 5, &vertexDescriptorBufferInfo);
	// 6
	VkDescriptorBufferInfo indexDescriptorBufferInfo{};
	indexDescriptorBufferInfo.buffer = indexBuffer.buffer;
	indexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	writeDescriptorSets[6] = descriptorSetContainer.makeWrite(0, 6, &indexDescriptorBufferInfo);
	vkUpdateDescriptorSets(context,                                            // The context
		static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
		writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
		0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)

	//// Write values into the descriptor set for Phase 1
	//std::array<VkWriteDescriptorSet, 1> generateWriteDescriptorSets;
	//// 0 - Ray buffer
	//VkDescriptorBufferInfo rayDescriptorBufferInfo{};
	//rayDescriptorBufferInfo.buffer = rayBuffer.buffer;
	//rayDescriptorBufferInfo.range = rayBufferSizeBytes;
	//generateWriteDescriptorSets[0] = generateDescriptorSetContainer.makeWrite(0, 0, &rayDescriptorBufferInfo);
	//vkUpdateDescriptorSets(context,
	//                       static_cast<uint32_t>(generateWriteDescriptorSets.size()),
	//                       generateWriteDescriptorSets.data(),
	//                       0, nullptr);


	////// Shader loading and pipeline creation
	//VkShaderModule rayTraceModule =
	//nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.comp.glsl.spv", true, searchPaths));
	// Load separate compute shaders for each phase
	VkShaderModule modules[] = {
	  nvvk::createShaderModule(context, nvh::loadFile("shaders/generate.comp.glsl.spv", true, searchPaths)),
	  nvvk::createShaderModule(context, nvh::loadFile("shaders/extend.comp.glsl.spv", true, searchPaths)),
	  nvvk::createShaderModule(context, nvh::loadFile("shaders/shade.comp.glsl.spv", true, searchPaths)),
	  nvvk::createShaderModule(context, nvh::loadFile("shaders/connect.comp.glsl.spv", true, searchPaths)),
	};

	//// Describes the entrypoint and the stage to use for this shader module in the pipeline
	//VkPipelineShaderStageCreateInfo shaderStageCreateInfo = nvvk::make<VkPipelineShaderStageCreateInfo>();
	//shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
	//shaderStageCreateInfo.module                          = rayTraceModule;
	//shaderStageCreateInfo.pName                           = "main";

	//// Create the compute pipeline
	//VkComputePipelineCreateInfo pipelineCreateInfo = nvvk::make<VkComputePipelineCreateInfo>();
	//pipelineCreateInfo.stage                       = shaderStageCreateInfo;
	//pipelineCreateInfo.layout                      = descriptorSetContainer.getPipeLayout();
	//// Don't modify flags, basePipelineHandle, or basePipelineIndex
	//VkPipeline computePipeline;
	//NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
	//                                    VK_NULL_HANDLE,          // Pipeline cache (uses default)
	//                                    1, &pipelineCreateInfo,  // Compute pipeline create info
	//                                    nullptr,                 // Allocator (uses default)
	//                                    &computePipeline));      // Output
	// Create separate pipelines for each phase
	VkPipeline pipelines[4];
	for (int i = 0; i < 4; ++i)
	{
		pipelines[i] = CreateComputePipeline(context, modules[i], descriptorSetContainer.getLayout());
	}

	// Create and start recording a command buffer
	VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

	//// Bind the compute shader pipeline
	//vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
	//// Bind the descriptor set
	//VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
	//vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
	//                        &descriptorSet, 0, nullptr);

	// iterations work as aa
	for (int iterations = 0; iterations < max_iterations; ++iterations)
	{
		// Loop through the phases and run the compute shaders
		for (int phase = 0; phase < 4; ++phase)
		{
			// Bind the appropriate pipeline for the current phase
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[phase]);
			// Bind the descriptor set
			VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(phase);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
				&descriptorSet, 0, nullptr);

			// Run the compute shader for the current phase
			vkCmdDispatch(cmdBuffer, (uint32_t(render_width) + workgroup_width - 1) / workgroup_width,
				(uint32_t(render_height) + workgroup_height - 1) / workgroup_height, 1);

			// End and submit the command buffer for Phase 1
			EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);

			switch (phase)
			{
				case 0:
				{
					void* generateMappedData = allocator.map(generateBuffer);
					// reinterpret the buffer as vec3
					Ray* generateBufferData = reinterpret_cast<Ray*>(generateMappedData);
					// Create an array of RayData
					std::vector<Ray> rayDataArray(render_height * render_width);

					// Copy the data from the mapped region to the array
					for (uint32_t i = 0; i < render_height * render_width; ++i)
					{
						rayDataArray[i].origin = glm::vec3(generateBufferData[i].origin);
						rayDataArray[i].direction = glm::vec3(generateBufferData[i].direction);
					}


					// Start a command buffer for uploading the buffers
					VkCommandBuffer uploadCmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
					// We get these buffers' device addresses, and use them as storage buffers and build inputs.
					const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
						| VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
					generateBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);

					EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
					allocator.finalizeAndReleaseStaging();
					break;
				}
				case 1: 
				{
					void* extendMappedData = allocator.map(extendBuffer);
					// reinterpret the buffer as vec3
					Ray* extendBufferData = reinterpret_cast<Ray*>(extendMappedData);
					break;
				}
				case 2: 
				{
					break;
				}
				case 3: 
				{
					break;
				}
			}
		}
	}
	allocator.unmap(generateBuffer);
	allocator.unmap(extendBuffer);

	// Run the compute shader with enough workgroups to cover the entire buffer:
	vkCmdDispatch(cmdBuffer, (uint32_t(render_width) + workgroup_width - 1) / workgroup_width,
		(uint32_t(render_height) + workgroup_height - 1) / workgroup_height, 1);

	// Add a command that says "Make it so that memory writes by the compute shader
	// are available to read from the CPU." (In other words, "Flush the GPU caches
	// so the CPU can read the data.") To do this, we use a memory barrier.
	// This is one of the most complex parts of Vulkan, so don't worry if this is
	// confusing! We'll talk about pipeline barriers more in the extras.
	VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
	memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;  // Make shader writes
	memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;     // Readable by the CPU
	vkCmdPipelineBarrier(cmdBuffer,                              // The command buffer
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // From the compute shader
		VK_PIPELINE_STAGE_HOST_BIT,             // To the CPU
		0,                                      // No special flags
		1, &memoryBarrier,                      // An array of memory barriers
		0, nullptr, 0, nullptr);                // No other barriers

	// End and submit the command buffer, then wait for it to finish:
	EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);

	//// Get the image data back from the GPU
	//void* data = allocator.map(buffer);
	//stbi_write_hdr("out.hdr", render_width, render_height, 3, reinterpret_cast<float*>(data));
	//allocator.unmap(buffer);

	//vkDestroyPipeline(context, computePipeline, nullptr);
	//vkDestroyShaderModule(context, rayTraceModule, nullptr);
	// Cleanup
	for (int i = 0; i < 4; ++i)
	{
		vkDestroyPipeline(context, pipelines[i], nullptr);
		vkDestroyShaderModule(context, modules[i], nullptr);
	}

	raytracingBuilder.destroy();
	allocator.destroy(vertexBuffer);
	allocator.destroy(indexBuffer);
	vkDestroyCommandPool(context, cmdPool, nullptr);
	//allocator.destroy(buffer);
	allocator.deinit();
	context.deinit();  // Don't forget to clean up at the end of the program!
}
