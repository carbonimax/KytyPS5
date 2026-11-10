#include "graphics/shader/stubStageShader.h"

#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"

#include <string>
#include <unordered_map>
#include <utility>

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::Spirv::Builder;

struct SourceModule {
	uint32_t                                                version    = 0;
	uint32_t                                                addressing = 0;
	uint32_t                                                memory     = 0;
	std::vector<uint32_t>                                   capabilities;
	std::vector<std::string>                                extensions;
	std::unordered_map<uint32_t, std::span<const uint32_t>> definitions; // types and constants
	std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> decorations;
	struct Variable {
		uint32_t id;
		uint32_t pointer_type;
		uint32_t storage_class;
	};
	std::vector<Variable> variables;
};

bool Parse(std::span<const uint32_t> words, SourceModule& out) {
	if (words.size() < 5 || words[0] != spv::MagicNumber) {
		return false;
	}
	out.version = words[1];
	for (size_t at = 5; at < words.size();) {
		const auto count = words[at] >> spv::WordCountShift;
		const auto op    = words[at] & spv::OpCodeMask;
		if (count == 0 || at + count > words.size()) {
			return false;
		}
		const auto inst = words.subspan(at, count);
		switch (op) {
			case spv::OpCapability: out.capabilities.push_back(inst[1]); break;
			case spv::OpExtension:
				out.extensions.emplace_back(reinterpret_cast<const char*>(inst.data() + 1));
				break;
			case spv::OpMemoryModel:
				out.addressing = inst[1];
				out.memory     = inst[2];
				break;
			case spv::OpDecorate:
				out.decorations[inst[1]].emplace_back(inst[2], count > 3 ? inst[3] : 0u);
				break;
			case spv::OpTypeVoid:
			case spv::OpTypeBool:
			case spv::OpTypeInt:
			case spv::OpTypeFloat:
			case spv::OpTypeVector:
			case spv::OpTypeArray:
			case spv::OpTypeStruct:
			case spv::OpTypePointer: out.definitions[inst[1]] = inst; break;
			case spv::OpConstant: out.definitions[inst[2]] = inst; break;
			case spv::OpVariable: out.variables.push_back({inst[2], inst[1], inst[3]}); break;
			case spv::OpFunction: return true; // Globals end here.
			default: break;
		}
		at += count;
	}
	return true;
}

// The source type `id` declared in `builder`; 0 when it is not a scalar, vector or array of them.
uint32_t ImportType(const SourceModule& source, Builder& builder, uint32_t id) {
	const auto found = source.definitions.find(id);
	if (found == source.definitions.end()) {
		return 0;
	}
	const auto inst = found->second;
	switch (inst[0] & spv::OpCodeMask) {
		case spv::OpTypeInt: return builder.Type(spv::OpTypeInt, inst[2], inst[3]);
		case spv::OpTypeFloat: return builder.Type(spv::OpTypeFloat, inst[2]);
		case spv::OpTypeVector: {
			const auto component = ImportType(source, builder, inst[2]);
			return component != 0 ? builder.Type(spv::OpTypeVector, component, inst[3]) : 0;
		}
		case spv::OpTypeArray: {
			const auto element = ImportType(source, builder, inst[2]);
			const auto length  = source.definitions.find(inst[3]);
			if (element == 0 || length == source.definitions.end() ||
			    (length->second[0] & spv::OpCodeMask) != spv::OpConstant || length->second.size() != 4) {
				return 0;
			}
			const auto length_type = ImportType(source, builder, length->second[1]);
			if (length_type == 0) {
				return 0;
			}
			return builder.Type(spv::OpTypeArray, element,
			                    builder.Constant(spv::OpConstant, length_type, length->second[3]));
		}
		default: return 0;
	}
}

// A `model` stub whose outputs copy the Location variables of `storage_class` in `words`.
std::vector<uint32_t> BuildStub(std::span<const uint32_t> words, spv::StorageClass storage_class,
                                spv::ExecutionModel model) {
	SourceModule source;
	if (!Parse(words, source)) {
		return {};
	}
	Builder builder(source.version);
	for (const auto capability: source.capabilities) {
		builder.RequireCapability(static_cast<spv::Capability>(capability));
	}
	for (const auto& extension: source.extensions) {
		builder.RequireExtension(extension.c_str());
	}
	builder.AddMemoryModel(static_cast<spv::AddressingModel>(source.addressing),
	                       static_cast<spv::MemoryModel>(source.memory));
	const auto void_type     = builder.Type(spv::OpTypeVoid);
	const auto function_type = builder.Type(spv::OpTypeFunction, void_type);

	std::vector<uint32_t>                      interfaces;
	std::vector<std::pair<uint32_t, uint32_t>> stores; // variable, type
	const auto output = [&](uint32_t type) {
		const auto variable = builder.DefineGlobalVariable(
		    builder.Type(spv::OpTypePointer, spv::StorageClassOutput, type), spv::StorageClassOutput);
		interfaces.push_back(variable);
		stores.emplace_back(variable, type);
		return variable;
	};
	for (const auto& variable: source.variables) {
		if (variable.storage_class != static_cast<uint32_t>(storage_class)) {
			continue;
		}
		const auto decorations = source.decorations.find(variable.id);
		if (decorations == source.decorations.end()) {
			continue;
		}
		bool location = false, builtin = false, per_vertex = false;
		for (const auto& [decoration, literal]: decorations->second) {
			location |= decoration == spv::DecorationLocation;
			builtin |= decoration == spv::DecorationBuiltIn;
			per_vertex |= decoration == spv::DecorationPerVertexKHR;
		}
		if (!location || builtin) {
			continue;
		}
		const auto pointer = source.definitions.find(variable.pointer_type);
		if (pointer == source.definitions.end()) {
			return {};
		}
		auto pointee = pointer->second[3];
		if (per_vertex) {
			// A per-vertex PS input reads the plain value the vertex stage writes.
			const auto array = source.definitions.find(pointee);
			if (array == source.definitions.end() ||
			    (array->second[0] & spv::OpCodeMask) != spv::OpTypeArray) {
				return {};
			}
			pointee = array->second[2];
		}
		const auto type = ImportType(source, builder, pointee);
		if (type == 0) {
			return {};
		}
		const auto copy = output(type);
		for (const auto& [decoration, literal]: decorations->second) {
			const bool keep = decoration == spv::DecorationLocation ||
			                  decoration == spv::DecorationComponent ||
			                  (decoration == spv::DecorationIndex && model == spv::ExecutionModelFragment);
			if (keep) {
				builder.AddAnnotation(spv::OpDecorate, copy, static_cast<spv::Decoration>(decoration),
				                      literal);
			}
		}
	}
	if (model == spv::ExecutionModelVertex) {
		const auto vec4     = builder.Type(spv::OpTypeVector, builder.Type(spv::OpTypeFloat, 32u), 4u);
		const auto position = output(vec4);
		builder.AddAnnotation(spv::OpDecorate, position, spv::DecorationBuiltIn, spv::BuiltInPosition);
	}
	std::vector<uint32_t> zeros;
	zeros.reserve(stores.size());
	for (const auto& store: stores) {
		zeros.push_back(builder.Constant(spv::OpConstantNull, store.second));
	}

	const auto main = builder.AllocateId();
	builder.AddEntryPoint(model, main, "main", interfaces);
	if (model == spv::ExecutionModelFragment) {
		builder.AddExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
	}
	builder.AddFunction(spv::OpFunction, void_type, main, spv::FunctionControlMaskNone, function_type);
	builder.AddFunction(spv::OpLabel, builder.AllocateId());
	for (size_t i = 0; i < stores.size(); i++) {
		builder.AddFunction(spv::OpStore, stores[i].first, zeros[i]);
	}
	builder.AddFunction(spv::OpReturn);
	builder.AddFunction(spv::OpFunctionEnd);
	return builder.Build();
}

} // namespace

std::vector<uint32_t> BuildStubVertexForPixel(std::span<const uint32_t> pixel) {
	return BuildStub(pixel, spv::StorageClassInput, spv::ExecutionModelVertex);
}

std::vector<uint32_t> BuildStubVertexLike(std::span<const uint32_t> vertex) {
	return BuildStub(vertex, spv::StorageClassOutput, spv::ExecutionModelVertex);
}

std::vector<uint32_t> BuildStubPixelLike(std::span<const uint32_t> pixel) {
	return BuildStub(pixel, spv::StorageClassOutput, spv::ExecutionModelFragment);
}

} // namespace Libs::Graphics
