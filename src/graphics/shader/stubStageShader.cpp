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

// The source type `id` filled with `index` (a signed 32-bit value) converted to each component.
uint32_t VaryingValue(const SourceModule& source, Builder& builder, uint32_t id, uint32_t index) {
	const auto inst = source.definitions.at(id);
	if ((inst[0] & spv::OpCodeMask) == spv::OpTypeInt && inst[2] == 32u && inst[3] == 1u) {
		return index;
	}
	const auto type  = ImportType(source, builder, id);
	const auto value = builder.AllocateId();
	switch (inst[0] & spv::OpCodeMask) {
		case spv::OpTypeFloat: builder.AddFunction(spv::OpConvertSToF, type, value, index); break;
		case spv::OpTypeInt:
			if (inst[2] != 32u) {
				builder.AddFunction(spv::OpSConvert, type, value, index);
			} else {
				builder.AddFunction(spv::OpBitcast, type, value, index);
			}
			break;
		default: { // Vector or array of a converted scalar.
			const auto element = VaryingValue(source, builder, inst[2], index);
			const auto count   = (inst[0] & spv::OpCodeMask) == spv::OpTypeVector
			                         ? inst[3]
			                         : source.definitions.at(inst[3])[3];
			const std::vector<uint32_t> elements(count, element);
			builder.AddFunction(spv::OpCompositeConstruct, type, value, std::span<const uint32_t>(elements));
			break;
		}
	}
	return value;
}

// A `model` stub whose outputs copy the Location variables of `storage_class` in `words`.
// `varying`: the outputs hold the vertex index instead of zero (no constant to fold).
std::vector<uint32_t> BuildStub(std::span<const uint32_t> words, spv::StorageClass storage_class,
                                spv::ExecutionModel model, bool varying = false) {
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
	std::vector<std::pair<uint32_t, uint32_t>> stores;        // variable, type
	std::vector<uint32_t>                      source_types;  // the store's source type, 0 for the position
	const auto output = [&](uint32_t type, uint32_t source_type) {
		const auto variable = builder.DefineGlobalVariable(
		    builder.Type(spv::OpTypePointer, spv::StorageClassOutput, type), spv::StorageClassOutput);
		interfaces.push_back(variable);
		stores.emplace_back(variable, type);
		source_types.push_back(source_type);
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
		const auto copy = output(type, pointee);
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
		const auto position = output(vec4, 0);
		builder.AddAnnotation(spv::OpDecorate, position, spv::DecorationBuiltIn, spv::BuiltInPosition);
	}
	std::vector<uint32_t> zeros;
	zeros.reserve(stores.size());
	for (const auto& store: stores) {
		zeros.push_back(builder.Constant(spv::OpConstantNull, store.second));
	}
	const auto int_type     = builder.Type(spv::OpTypeInt, 32u, 1u);
	const auto vertex_index = varying ? builder.DefineGlobalVariable(
	                                        builder.Type(spv::OpTypePointer, spv::StorageClassInput, int_type),
	                                        spv::StorageClassInput)
	                                  : 0u;
	if (varying) {
		interfaces.push_back(vertex_index);
		builder.AddAnnotation(spv::OpDecorate, vertex_index, spv::DecorationBuiltIn, spv::BuiltInVertexIndex);
	}

	const auto main = builder.AllocateId();
	builder.AddEntryPoint(model, main, "main", interfaces);
	if (model == spv::ExecutionModelFragment) {
		builder.AddExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
	}
	builder.AddFunction(spv::OpFunction, void_type, main, spv::FunctionControlMaskNone, function_type);
	builder.AddFunction(spv::OpLabel, builder.AllocateId());
	const auto index = varying ? builder.AllocateId() : 0u;
	if (varying) {
		builder.AddFunction(spv::OpLoad, int_type, index, vertex_index);
	}
	for (size_t i = 0; i < stores.size(); i++) {
		auto value = zeros[i];
		if (varying && source_types[i] != 0) {
			value = VaryingValue(source, builder, source_types[i], index);
		} else if (varying) { // The position.
			const auto f32 = builder.Type(spv::OpTypeFloat, 32u);
			const auto x   = builder.AllocateId();
			builder.AddFunction(spv::OpConvertSToF, f32, x, index);
			value = builder.AllocateId();
			const std::vector<uint32_t> elements(4, x);
			builder.AddFunction(spv::OpCompositeConstruct, stores[i].second, value,
			                    std::span<const uint32_t>(elements));
		}
		builder.AddFunction(spv::OpStore, stores[i].first, value);
	}
	builder.AddFunction(spv::OpReturn);
	builder.AddFunction(spv::OpFunctionEnd);
	return builder.Build();
}

} // namespace

std::vector<uint32_t> BuildStubVertexForPixel(std::span<const uint32_t> pixel) {
	return BuildStub(pixel, spv::StorageClassInput, spv::ExecutionModelVertex);
}

std::vector<uint32_t> BuildStubVaryingVertexForPixel(std::span<const uint32_t> pixel) {
	return BuildStub(pixel, spv::StorageClassInput, spv::ExecutionModelVertex, true);
}

std::vector<uint32_t> BuildStubVertexLike(std::span<const uint32_t> vertex) {
	return BuildStub(vertex, spv::StorageClassOutput, spv::ExecutionModelVertex);
}

std::vector<uint32_t> BuildStubPixelLike(std::span<const uint32_t> pixel) {
	return BuildStub(pixel, spv::StorageClassOutput, spv::ExecutionModelFragment);
}

} // namespace Libs::Graphics
