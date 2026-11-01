#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "SrtWalkerTestMemory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

// Every extracted plan below is the one a restarted game loads from the shader cache: exactly
// its bytes load, and the loaded plan saves back to them.
Libs::Graphics::ShaderRecompiler::IR::ResourcePlan
Extract(const Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto bytes = SaveResourcePlan(ExtractResourcePlan(program));
  auto extended = bytes;
  extended.push_back(0u);
  ResourcePlan loaded, truncated, padded;
  Check(LoadResourcePlan(bytes, loaded) && SaveResourcePlan(loaded) == bytes &&
            !LoadResourcePlan(std::span(bytes).first(bytes.size() - 1u), truncated) &&
            !LoadResourcePlan(extended, padded),
        "resource plan did not round-trip through its exact cache bytes");
  return loaded;
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
  program.block_storage.push_back(std::move(block));
  return *result;
}

// Lists one extra source per block and walks the plan: each source's activity is its block's reachability.
std::vector<uint8_t> ReachedBlocks(Libs::Graphics::ShaderRecompiler::IR::ResourcePlan &plan,
                                   const Libs::Graphics::ShaderRecompiler::IR::SrtRuntime &runtime) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto first = static_cast<uint32_t>(plan.descriptor_sources.size());
  for (uint32_t block = 0; block < plan.control_flow.size(); ++block) {
    plan.descriptor_sources.emplace_back();
    plan.control_flow[block].sources.push_back(first + block);
  }
  CompileSrtWalker(plan);
  std::vector<uint32_t> flat;
  Check(WalkSrt(plan, runtime, flat, nullptr), "reachability walk failed");
  std::vector<uint8_t> reached;
  for (uint32_t block = 0; block < plan.control_flow.size(); ++block)
    reached.push_back(WalkedActive(plan, first + block));
  return reached;
}

Libs::Graphics::ShaderRecompiler::IR::Program SrtProgram(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return program;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return Extract(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return Extract(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestWalkerLoadsBypassSpecializationReader() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = Extract(SrtProgram(reinterpret_cast<uint64_t>(&dword)));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "walker load used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "walker did not load the mapped SRT DWORD");
}

void TestNativeNestedSrtRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::array<uint32_t, 3> first{0u, 0x12345678u, 0xabcdef01u};
  std::array<uint32_t, 3> second{0u, 0x87654321u, 0x10fedcbau};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address) | 3u,
                                  static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
  };
  auto table = pointer_words(first.data());
  const auto user_data = pointer_words(table.data());
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 2;
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto low = emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))});
  const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto read = [&](Value address, uint32_t offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {address, Value(offset), Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto pointer_low = read(root, 0u);
  const auto pointer_high = read(root, 4u);
  auto *leaf = program.srt_reads[0].value.Instruction();
  leaf->Arg(0).Instruction()->SetArg(0, pointer_low);
  leaf->Arg(0).Instruction()->SetArg(1, pointer_high);
  leaf->SetArg(1, Value(7u));
  read(leaf->Arg(0), 11u);
  auto plan = Extract(program);
  Check(plan.srt_code != nullptr, "nested user-data tree did not compile an x86 walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const SrtRuntime runtime{.user_data = user_data};
  for (const auto *data : {&first, &second}) {
    table = pointer_words(data->data());
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt ==
                  std::vector<uint32_t>{(*data)[1], table[0], table[1], (*data)[2]},
          "native SRT refresh lost pointer masking, alignment, or current values");
  }
  {
    // Unmapped tables and the null pointer they yield read as zero.
    FakeGuestMemory unmapped;
    const std::array<uint32_t, 2> unmapped_table{0x1000u, 0u};
    Check(MaterializeResources(plan, {.user_data = unmapped_table}, snapshot, specialization) &&
              snapshot.flattened_srt == std::vector<uint32_t>(4, 0u),
          "faulting walker loads of a table and a null pointer did not read zero");
  }
  Check(!MaterializeResources(plan, {.user_data = std::span(user_data).first(1)},
                              snapshot, specialization),
        "native SRT walk accepted truncated user data");
  const std::array<uint32_t, 2> fake_user_data{0x1003u, 0xabcd0000u};
  uint32_t reads = 0;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++*static_cast<uint32_t *>(data);
        switch (address) {
          case 0x1000u: words[0] = 0x2003u; break;
          case 0x1004u: words[0] = 0xabcd0000u; break;
          case 0x2004u: words[0] = 0x1234u; break;
          case 0x2008u: words[0] = 0x5678u; break;
          default: return false;
        }
        return true;
      }, &reads);
  Check(MaterializeResources(plan, {.user_data = fake_user_data}, snapshot, specialization) && reads == 4 &&
            snapshot.flattened_srt ==
                std::vector<uint32_t>{0x1234u, 0x2003u, 0xabcd0000u, 0x5678u},
        "native SRT walk repeated nested pointer reads");
  leaf->SetArg(0, root);
  leaf->SetArg(1, Value(0u));
  program.srt_reads.resize(1);
  for (const int32_t offset : {-1, 7}) {
    program.memory_info[0].offset = static_cast<uint32_t>(offset);
    auto bounded = Extract(program);
    auto base = pointer_words(second.data() + 1);
    Check(bounded.srt_code != nullptr &&
              MaterializeResources(bounded, {.user_data = base}, snapshot, specialization) &&
              snapshot.flattened_srt[0] == second[offset < 0 ? 0 : 2],
          "native SRT applied a signed immediate offset incorrectly");
    base = offset < 0 ? std::array<uint32_t, 2>{0u, 0u}
                      : std::array<uint32_t, 2>{0xfffffffcu, 0xffffu};
    Check(!MaterializeResources(bounded, {.user_data = base}, snapshot, specialization),
          "native SRT accepted address underflow or 48-bit overflow");
  }
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtProgram(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestSrtAliasesShareReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = SrtProgram(0x1000u);
  auto &block = *program.blocks[0];
  const auto append_read = [&](uint32_t address) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                       {Value(address), Value(0u)});
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(0u), Value(0u), Value(true)});
    // Different reads deliberately share MemoryInfo but have different guest PCs.
    read.SetFlags(MemoryFlags{.index = 0, .pc = address});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&read), slot});
    auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto ordinary = append_read(0x2000u);
  const auto pointer = append_read(0x3000u);
  program.srt_reads[0].value.Instruction()->Arg(0).Instruction()->SetArg(0, pointer);
  auto &mask = block.AppendNewInst(ValueOpcode::INotEqual32, {ordinary, Value(0u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                    {Value(9u), Value(&mask)});
  program.descriptor_sources.push_back({.dwords = {Value(&first), Value(0u)},
                                       .dword_count = 2});
  DescriptorSource indirect;
  indirect.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1};
  program.descriptor_sources.push_back(indirect);
  auto plan = Extract(program);
  for (const auto &inst : plan.value_storage) {
    Check(inst.GetOpcode() != ValueOpcode::ReadConst &&
              inst.GetOpcode() != ValueOpcode::GetSrtResource,
          "retained plan still contains flattened-read aliases");
  }
  Check(plan.descriptor_sources[0].dwords[0] == plan.srt_reads[0].value,
        "alias normalization lost read identity");
  struct Reads { uint32_t count = 0; bool dirty = false; } reads;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.count;
        if (address == 0x1000u) words[0] = 0x1111u;
        else if (address == 0x2000u) words[0] = 0x2222u;
        else if (address == 0x3000u && !reads.dirty) words[0] = 0x1000u;
        else return false;
        return true;
      }, &reads);
  std::vector<uint32_t> flat;
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
            reads.count == 3,
        "normalized aliases changed nested reads or repeated a shared read");
  // An unreadable pointer reads as zero, and so does the null table it yields.
  reads = {.dirty = true};
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0u, 0x2222u, 0u} &&
            reads.count == 3,
        "unreadable pointer did not read as zero");
}

void TestProtectedPageLoadsCallReader() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = SrtProgram(0x1000u);
  auto &block = *program.blocks[0];
  const auto append_read = [&](uint32_t address) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                       {Value(address), Value(0u)});
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(0u), Value(0u), Value(true)});
    // Different reads deliberately share MemoryInfo but have different guest PCs.
    read.SetFlags(MemoryFlags{.index = 0, .pc = address});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&read), slot});
    auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto ordinary = append_read(0x2000u);
  const auto pointer = append_read(0x3000u);
  program.srt_reads[0].value.Instruction()->Arg(0).Instruction()->SetArg(0, pointer);
  auto &mask = block.AppendNewInst(ValueOpcode::INotEqual32, {ordinary, Value(0u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                    {Value(9u), Value(&mask)});
  program.descriptor_sources.push_back({.dwords = {Value(&first), Value(0u)},
                                       .dword_count = 2});
  DescriptorSource indirect;
  indirect.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1};
  program.descriptor_sources.push_back(indirect);
  auto plan = ExtractResourcePlan(program);
  for (const auto &inst : plan.value_storage) {
    Check(inst.GetOpcode() != ValueOpcode::ReadConst &&
              inst.GetOpcode() != ValueOpcode::GetSrtResource,
          "retained plan still contains flattened-read aliases");
  }
  Check(plan.descriptor_sources[0].dwords[0] == plan.srt_reads[0].value,
        "alias normalization lost read identity");
  struct Reads { uint32_t count = 0; bool dirty = false; } reads;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.count;
        if (address == 0x1000u) words[0] = 0x1111u;
        else if (address == 0x2000u) words[0] = 0x2222u;
        else if (address == 0x3000u && !reads.dirty) words[0] = 0x1000u;
        else return false;
        return true;
      }, &reads);
  std::vector<uint32_t> flat;
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
            reads.count == 3,
        "normalized aliases changed nested reads or repeated a shared read");
  // An unreadable pointer reads as zero, and so does the null table it yields.
  reads = {.dirty = true};
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0u, 0x2222u, 0u} &&
            reads.count == 3,
        "unreadable pointer did not read as zero");
  // A load from a page listed as protected calls the reader instead of faulting; a failed reader
  // call falls back to the direct load (here: the fault path).
  static uint32_t protected_calls = 0;
  protected_calls = 0;
  SetProtectedLoadReader(+[](uint64_t address, uint32_t *value) -> uint32_t {
    ++protected_calls;
    if (address != 0x2000u) return ProtectedLoadKeep;
    *value = 0x7777u;
    return ProtectedLoadValue | ProtectedLoadKeep;
  });
  NoteProtectedLoad(0x2000u);
  NoteProtectedLoad(0x3000u);
  reads = {};
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0x1111u, 0x7777u, 0x1000u} &&
            reads.count == 2 && protected_calls == 2,
        "protected page loads did not go through the reader");
  SetProtectedLoadReader(nullptr);
  reads = {};
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
            reads.count == 3,
        "protected page loads without a reader did not fall back to the direct load");
  // A page the reader reports unprotected leaves the list: the next walk loads it directly.
  protected_calls = 0;
  SetProtectedLoadReader(+[](uint64_t, uint32_t *) -> uint32_t {
    ++protected_calls;
    return 0;
  });
  NoteProtectedLoad(0x2000u);
  reads = {};
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
            protected_calls == 2,
        "unprotected pages were not read directly");
  reads = {};
  Check(WalkSrt(plan, {}, flat, nullptr) && flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
            protected_calls == 2 && reads.count == 3,
        "unprotected pages stayed listed");
  SetProtectedLoadReader(nullptr);
}

void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t count = 0; bool readable = true; } reads;
  FakeGuestMemory fake([](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.count;
        if (!reads.readable || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }, &reads);
  const SrtRuntime runtime;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = Extract(program);
    Check(plan.control_flow.empty() && plan.capture_specialization_reads,
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.count == 1 &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "vector descriptor input was not read and captured exactly once");
    // An unreadable input reads as zero, so the count selects 0 + 1.
    reads = {.readable = false};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 1 && reads.count == 1,
          "unreadable vector descriptor input did not read as zero");
  }
  // A read guarded by its EXEC mask depends on that mask, so it cannot reuse the unmasked read.
  auto &guarded = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(&active)});
  guarded.SetFlags(MemoryFlags{.index = 0});
  auto &guarded_lane = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&guarded), Value(&active)});
  program.descriptor_sources.push_back({.dwords = {Value(&guarded_lane)}, .dword_count = 1});
  {
    auto plan = Extract(program);
    uint32_t value = 0;
    reads = {};
    Check(EvaluateSrtValue(plan, runtime, plan.descriptor_sources.back().dwords[0], value) &&
              value == 72u && reads.count == 1,
          "EXEC-guarded vector read lost its mask");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  program.descriptor_sources.push_back({.dwords = {Value(&inactive)}, .dword_count = 1});
  reads.count = 0;
  uint32_t result = 99;
  {
    auto plan = Extract(program);
    Check(EvaluateSrtValue(plan, runtime, plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.count == 0,
          "literal false EXEC read vector memory");
  }
  count.SetArg(4, Value(true));
  handle.SetArg(3, Value(0x204u));
  program.descriptor_sources.back().dwords[0] = Value(&count);
  {
    auto plan = Extract(program);
    Check(EvaluateSrtValue(plan, runtime, plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.count == 0,
          "invalid vector buffer format read memory");
  }
  handle.SetArg(3, Value(0x16204u));
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(EvaluateSrtValue(program, {}, Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!EvaluateSrtValue(program, {}, Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorPredicateReads(bool memory_condition) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // Only memory-derived host decisions capture their descriptor reads.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
      {memory_condition ? Value(&read) : Value(&offset),
       Value(memory_condition ? 0x8000u : 4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.blocks[0]->condition = Value(&condition);
  program.blocks[0]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.blocks[0]->terminator.true_block = program.blocks[1];
  program.blocks[0]->terminator.false_block = program.blocks[2];
  program.blocks[1]->id = 1;
  program.blocks[1]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.blocks[2]->id = 2;
  program.blocks[2]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = Extract(program);
  Check(plan.capture_specialization_reads == memory_condition && !plan.control_flow.empty(),
        "conditional writable descriptor used the wrong memory dependency policy");
  struct Reads { uint32_t count = 0; bool readable = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.count;
        if (!reads.readable || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }, &reads);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool readable : {false, true}) {
    reads = {.readable = readable};
    // The predicate and the descriptor share one read. An unreadable one is zero, which
    // makes a memory predicate skip the store block and leave its descriptor inactive.
    Check(MaterializeResources(plan, {.user_data = user_data}, snapshot, specialization) &&
              reads.count == 1,
          "writable descriptor repeated or skipped its shared read");
    Check(snapshot.buffers[0].dwords[0] == (readable ? 0x8000u : 0u) &&
              snapshot.specialization_reads == (memory_condition
                  ? std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}}
                  : std::vector<std::pair<uint64_t, uint64_t>>{}),
          "writable descriptor lost its address or captured an immutable predicate");
  }
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestSharedControlPredicates() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 8;
  for (const auto operation : {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr}) {
    Program program;
    program.user_data_count = 1;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    auto &block = AddValueBlock(program);
    const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
      return Value(&block.AppendNewInst(opcode, args));
    };
    const auto user = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
    const auto uniform = emit(ValueOpcode::INotEqual32, {user, Value(0u)});
    const auto lane = emit(ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    auto varying = emit(ValueOpcode::INotEqual32, {lane, Value(0u)});
    // Both failed and partially uniform subgraphs share their operands exponentially.
    for (uint32_t i = 0; i < depth; ++i)
      varying = emit(ValueOpcode::LogicalOr, {varying, varying});
    auto condition = emit(operation, {varying, uniform});
    for (uint32_t i = 0; i < depth; ++i)
      condition = emit(ValueOpcode::LogicalOr, {condition, condition});
    condition = emit(ValueOpcode::LogicalNot, {condition});
    condition = emit(ValueOpcode::ConditionRef, {condition});
    for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
    for (uint32_t i = 0; i < 2; ++i) {
      program.blocks[i]->condition = condition;
      program.blocks[i]->terminator.kind = TerminatorKind::ConditionalBranch;
      program.blocks[i]->terminator.true_block = program.blocks[i == 0 ? 1 : 3];
      program.blocks[i]->terminator.false_block = program.blocks[2];
    }
    auto plan = Extract(program);
    Check(plan.value_storage.size() <= 2u * depth + 12u &&
              plan.control_flow.size() == 4 &&
              plan.control_flow[0].condition == plan.control_flow[1].condition &&
              !plan.capture_specialization_reads,
          "shared predicates expanded into trees or retained varying dependencies");
    for (const uint32_t flag : {0u, 1u}) {
      const std::array<uint32_t, 1> user_data{flag};
      uint32_t result = 99;
      const bool conjunction = operation == ValueOpcode::LogicalAnd;
      const bool known = conjunction ? flag == 0 : flag != 0;
      Check(EvaluateSrtValue(plan, {.user_data = user_data}, plan.control_flow[0].condition, result) == known &&
                (!known || result == uint32_t{conjunction}),
            "shared predicate changed its uniform factor or cached runtime values");
      Check(ReachedBlocks(plan, {.user_data = user_data}) == (known
                    ? (conjunction ? std::vector<uint8_t>{1, 1, 0, 1}
                                   : std::vector<uint8_t>{1, 0, 1, 0})
                    : std::vector<uint8_t>{1, 1, 1, 1}),
            "shared predicate skipped an unknown branch or followed a rejected branch");
    }
  }
}

void TestNestedExecPredicatesReuseReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 12;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &use = *program.blocks[1];
  auto &exit = *program.blocks[2];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&entry.AppendNewInst(opcode, args));
  };
  const auto lane = emit(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  auto condition = emit(ValueOpcode::LogicalNot,
      {emit(ValueOpcode::ULessThanEqual32, {count, lane})});
  // A compact shared DAG must not become a tree of fresh EXEC evaluations.
  for (uint32_t i = 0; i < depth; ++i) {
    const auto mask = emit(ValueOpcode::INotEqual32, {lane, Value(i)});
    condition = emit(ValueOpcode::LogicalAnd,
        {mask, emit(ValueOpcode::LogicalOr, {condition, condition})});
  }
  entry.condition = condition;
  entry.terminator.kind = TerminatorKind::ConditionalBranch;
  entry.terminator.true_block = &use;
  entry.terminator.false_block = &exit;
  entry.AddBranch(&use);
  entry.AddBranch(&exit);
  use.terminator.kind = TerminatorKind::Branch;
  use.terminator.true_block = &exit;
  use.AddBranch(&exit);
  DescriptorSource source;
  source.dwords = {Value(0x8000u), Value(1u << 16u), Value(16u), Value(3u << 28u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = Value(&use.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
  use.AppendNewInst(ValueOpcode::LoadBufferU32,
      {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = Extract(program);
  struct Reads { uint32_t value = 0; bool readable = true; uint32_t count = 0; } reads;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.count;
        if (!reads.readable || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }, &reads);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{64u, true}, Reads{0u, true}, Reads{64u, false}, Reads{64u, true}}) {
    reads = state;
    Check(MaterializeResources(plan, {}, snapshot, specialization),
          "nested EXEC predicates rejected a valid resource snapshot");
    // An unreadable count reads as zero.
    const bool inactive = !reads.readable || reads.value == 0;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "nested EXEC predicates lost an unknown branch or retained stale runtime facts");
    Check(reads.count == 1 && snapshot.specialization_reads ==
              std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "EXEC contexts repeated a mask-independent scalar read");
  }
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = Extract(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
  } reads;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }, &reads);
  const SrtRuntime runtime;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    const auto offset = specialization.images[0].indirect_mapping_offset;
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated or skipped scalar reads");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
}

void TestEmptyTraversalSkipsUnmappedDescriptor(bool independent_mask, bool counted_loop = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 6; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &loop = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &merge = *program.blocks[3];
  auto &use = *program.blocks[4];
  auto &exit = *program.blocks[5];
  const auto emit = [](Block &block, ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto branch = [](Block &block, Block &target, Value condition = {}, Block *other = nullptr) {
    block.terminator.kind = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other) block.AddBranch(other);
  };
  Value lane, active, initial;
  if (!counted_loop) {
    lane = emit(entry, ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    active = emit(entry, ValueOpcode::INotEqual32, {lane, Value(0u)});
    const auto initial_mask = independent_mask
        ? emit(entry, ValueOpcode::IEqual32, {lane, Value(0u)}) : active;
    initial = emit(entry, ValueOpcode::SelectU32,
        {initial_mask, Value(UINT32_MAX), Value(0u)});
  }
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(entry, ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(entry, ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  if (counted_loop) {
    branch(entry, loop);
  } else {
    const auto rejected = emit(entry, ValueOpcode::ULessThanEqual32, {count, lane});
    const auto accepted = emit(entry, ValueOpcode::LogicalNot, {rejected});
    branch(entry, loop, emit(entry, ValueOpcode::LogicalAnd, {active, accepted}), &merge);
  }

  // A nonempty traversal contains a varying loop-carried value. Its cycle must
  // remain unknown; only the proven empty traversal can select the bypass arm.
  auto &carried = loop.AppendNewInst(ValueOpcode::Phi);
  carried.SetFlags(Type::U32);
  const auto next = emit(body, ValueOpcode::IAdd32, {Value(&carried), Value(1u)});
  carried.AddPhiOperand(&entry, counted_loop ? Value(0u) : lane);
  carried.AddPhiOperand(&body, next);
  const auto condition = counted_loop
      ? emit(loop, ValueOpcode::ULessThan32, {Value(&carried), count})
      : emit(loop, ValueOpcode::INotEqual32, {Value(&carried), Value(UINT32_MAX)});
  branch(loop, counted_loop ? use : body, condition, &merge);
  loop.terminator.loop_header = true;
  loop.terminator.continue_block = &body;
  loop.terminator.merge_block = &merge;
  branch(body, loop);
  if (counted_loop) {
    branch(merge, exit);
    branch(use, body);
  } else {
    auto &selected = merge.AppendNewInst(ValueOpcode::Phi);
    selected.SetFlags(Type::U32);
    selected.AddPhiOperand(&entry, initial);
    selected.AddPhiOperand(&loop, Value(&carried));
    const auto valid = emit(merge, ValueOpcode::INotEqual32, {Value(&selected), Value(UINT32_MAX)});
    branch(merge, use, emit(merge, ValueOpcode::LogicalAnd, {active, valid}), &exit);
    branch(use, exit);
  }
  DescriptorSource source;
  const auto descriptor_address = emit(use, ValueOpcode::GetAddressResource,
      {Value(0x2000u), Value(0u)});
  for (uint32_t i = 0; i < 4; ++i) {
    const auto index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
    source.dwords[i] = emit(use, ValueOpcode::LoadAddressU32,
        {descriptor_address, Value(i * 4u), Value(0u), Value(true)});
    source.dwords[i].Instruction()->SetFlags(MemoryFlags{.index = index});
  }
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  const auto buffer_index = static_cast<uint32_t>(program.memory_info.size());
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = emit(use, ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  emit(use, ValueOpcode::LoadBufferU32, {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .Instruction()->SetFlags(MemoryFlags{.index = buffer_index});
  auto plan = Extract(program);
  struct Reads { uint32_t count = 0; bool readable = true; uint32_t descriptors = 0; } reads;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        constexpr std::array<uint32_t, 4> descriptor{0x8000u, 1u << 16u, 16u,
                                                   (3u << 28u) | 0x4dfacu};
        auto &reads = *static_cast<Reads *>(data);
        if (address == 0x1000u && reads.readable) {
          words[0] = reads.count;
          return true;
        }
        if (address < 0x2000u || address >= 0x2010u || address % 4u || words.size() != 1)
          return false;
        ++reads.descriptors;
        words[0] = descriptor[(address - 0x2000u) / 4u];
        return true;
      }, &reads);
  const SrtRuntime runtime;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{0u, true}, Reads{4u, true}, Reads{4u, false}, Reads{0u, true}}) {
    reads = state;
    if (!counted_loop) {
      uint32_t condition = 99;
      Check(!EvaluateSrtValue(plan, runtime, plan.control_flow[3].condition, condition),
            "direct merge evaluation used missing or previous runtime reachability");
    }
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "partial traversal proof rejected a valid resource snapshot");
    // An unreadable count reads as zero.
    const bool inactive = !independent_mask && (!reads.readable || reads.count == 0u);
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "empty traversal kept its descriptor or an uncertain traversal lost it");
    Check(specialization.buffers[0].zero_stride_oob == inactive,
          "inactive traversal did not expose dead vector reads to specialization");
    Check(reads.descriptors == (inactive ? 0u : 4u),
          "inactive traversal read its descriptor or an uncertain traversal skipped it");
    if (inactive) {
      Check(!snapshot.specialization_reads.empty(), "empty traversal lost its captured scalar read");
      for (const auto &read : snapshot.specialization_reads)
        Check(read == std::pair<uint64_t, uint64_t>{0x1000u, 4u},
              "empty traversal proof captured an unrelated memory range");
    }
  }
}

void TestCyclicControlPredicatesRemainUnknown() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 4; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &left = *program.blocks[1];
  auto &right = *program.blocks[2];
  auto &exit = *program.blocks[3];
  const auto lane = Value(&entry.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)}));
  const auto u = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(0u)}));
  const auto v = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(1u)}));
  auto &p = left.AppendNewInst(ValueOpcode::Phi);
  auto &q = right.AppendNewInst(ValueOpcode::Phi);
  p.SetFlags(Type::U1);
  q.SetFlags(Type::U1);
  const auto a = Value(&left.AppendNewInst(ValueOpcode::LogicalAnd, {u, Value(&p)}));
  const auto b = Value(&right.AppendNewInst(ValueOpcode::LogicalAnd, {v, Value(&q)}));
  p.AddPhiOperand(&entry, Value(true));
  p.AddPhiOperand(&right, b);
  q.AddPhiOperand(&entry, Value(true));
  q.AddPhiOperand(&left, a);
  const auto branch = [](Block &block, Value condition, Block &yes, Block &no) {
    block.condition = condition;
    block.terminator.kind = TerminatorKind::ConditionalBranch;
    block.terminator.true_block = &yes;
    block.terminator.false_block = &no;
    block.AddBranch(&yes);
    block.AddBranch(&no);
  };
  // Two loop entries have no single header whose Phi can be discarded up front.
  branch(entry, u, left, right);
  branch(left, a, right, exit);
  branch(right, b, left, exit);
  auto plan = Extract(program);
  uint32_t value = 99;
  Check(!EvaluateSrtValue(plan, {}, plan.control_flow[1].condition, value) &&
            !EvaluateSrtValue(plan, {}, plan.control_flow[2].condition, value),
        "cyclic varying control acquired an unsupported uniform value");
  Check(ReachedBlocks(plan, {}) == std::vector<uint8_t>({1, 1, 1, 1}),
        "cyclic control lost a possible branch during reachability");
}

void TestBackEdgeReachesPhiPredecessor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 3;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 4; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &header = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &exit = *program.blocks[3];
  const auto branch = [](Block &block, Block &target, Value condition = {}, Block *other = nullptr) {
    block.terminator.kind = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other) block.AddBranch(other);
  };
  // header: p = Phi(entry: 0, body: 1); table[1] == p enters body, which returns to header. The
  // first pass evaluates p and loads table[1] before body is reached; the walk must revisit both,
  // capture the load once and reach exit, which loads table[0].
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  const auto table = Value(&entry.AppendNewInst(ValueOpcode::GetAddressResource,
      {Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))})),
       Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(1))}))}));
  const auto load = [&](Block &block, uint32_t offset) -> Inst & {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32, {table, Value(offset), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    return read;
  };
  auto &p = header.AppendNewInst(ValueOpcode::Phi);
  p.SetFlags(Type::U32);
  p.AddPhiOperand(&entry, Value(0u));
  p.AddPhiOperand(&body, Value(1u));
  branch(entry, header);
  branch(header, body, Value(&header.AppendNewInst(ValueOpcode::IEqual32, {Value(&load(header, 4u)), Value(&p)})),
         &exit);
  branch(body, header);
  exit.terminator.kind = TerminatorKind::Return;
  program.srt_reads.push_back({Value(&load(exit, 0u)), 0});
  exit.AppendNewInst(ValueOpcode::ReadConst,
                     {Value(&exit.AppendNewInst(ValueOpcode::GetSrtResource)), Value(0u)});
  auto plan = Extract(program);
  Check(plan.control_flow.size() == 4 && plan.control_phi_predecessors.size() == 1,
        "back-edge plan lost its control flow or Phi");
  plan.capture_specialization_reads = true;
  const std::array<uint32_t, 2> user_data{0x1000u, 0u};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  {
    // The table is unmapped: its loads read zero instead of faulting the walk.
    FakeGuestMemory unmapped;
    Check(MaterializeResources(plan, {.user_data = user_data}, snapshot, specialization) &&
              snapshot.flattened_srt == std::vector<uint32_t>{0u},
          "faulting load in a reachable block did not read zero");
  }
  for (uint32_t selected : {0u, 1u}) {
    FakeGuestMemory fake(+[](void *userdata, uint64_t address, std::span<uint32_t> words) {
          words[0] = address == 0x1000u ? 0x5678u : *static_cast<uint32_t *>(userdata);
          return address == 0x1000u || address == 0x1004u;
        }, &selected);
    Check(MaterializeResources(plan, {.user_data = user_data}, snapshot, specialization) &&
              snapshot.flattened_srt == std::vector<uint32_t>{0x5678u},
          "back edge to an evaluated Phi left its exit unreachable");
    Check(snapshot.specialization_reads ==
              std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}, {0x1000u, 4u}},
          "Phi restart kept the loads of the abandoned pass");
    Check(ReachedBlocks(plan, {.user_data = user_data}) ==
              (selected == 0u ? std::vector<uint8_t>{1, 1, 1, 1} : std::vector<uint8_t>{1, 1, 0, 1}),
          "Phi restart followed a rejected edge or missed a reachable one");
  }
}

void TestEmptyScalarBufferSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct Fixture {
    Program program;
    std::vector<std::pair<Inst *, Type>> uses;
    Inst *nonnull = nullptr;
  };
  const auto make_fixture = [] {
    Fixture fixture;
    auto &program = fixture.program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    program.user_data_count = 4u;
    auto &block = AddValueBlock(program);
    DescriptorSource source;
    source.dword_count = 4u;
    for (uint32_t word = 0; word < 4u; ++word)
      source.dwords[word] = Value(&block.AppendNewInst(ValueOpcode::GetUserData,
          {Value(static_cast<ScalarReg>(word))}));
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 0u, .read = true, .scalar = true});
    const auto table = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
    std::array<Value, 4> words;
    for (uint32_t word = 0; word < 4u; ++word) {
      program.memory_info.push_back({.kind = ResourceKind::ScalarBuffer, .resource = 0u,
                                     .offset = word * 4u});
      auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {table, Value(0u)});
      read.SetFlags(MemoryFlags{.index = word});
      words[word] = Value(&read);
    }
    const auto handle = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], words[3]}));
    for (const auto opcode : {ValueOpcode::ReadConstBuffer, ValueOpcode::LoadBufferU8,
                              ValueOpcode::LoadBufferU16, ValueOpcode::LoadBufferU32,
                              ValueOpcode::LoadBufferU32x2, ValueOpcode::LoadBufferU32x3,
                              ValueOpcode::LoadBufferU32x4}) {
      const auto index = static_cast<uint32_t>(program.memory_info.size());
      program.memory_info.push_back({.kind = ResourceKind::IndirectBuffer, .resource = UINT32_MAX,
          .data_dwords = BufferComponentCount(opcode),
          .data_bits = opcode == ValueOpcode::LoadBufferU8 ? 8u : opcode == ValueOpcode::LoadBufferU16 ? 16u : 32u,
          .formatted = opcode == ValueOpcode::LoadBufferU32});
      auto &read = opcode == ValueOpcode::ReadConstBuffer
          ? block.AppendNewInst(opcode, {handle, Value(0u)})
          : block.AppendNewInst(opcode, {handle, Value(0u), Value(0u), Value(0u), Value(true)});
      read.SetFlags(MemoryFlags{.index = index});
      fixture.uses.emplace_back(&block.AppendNewInst(ValueOpcode::Identity, {Value(&read)}), read.GetType());
    }
    const auto nonnull = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], Value(1u)}));
    auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {nonnull, Value(0u)});
    read.SetFlags(MemoryFlags{.index = 4u});
    fixture.nonnull = &block.AppendNewInst(ValueOpcode::Identity, {Value(&read)});
    return fixture;
  };
  auto fixture = make_fixture();
  auto plan = Extract(fixture.program);
  std::array<uint32_t, 4> descriptor{0x1000u, 8u << 16u, 0u, 3u << 28u};
  uint32_t reads = 0u;
  const SrtRuntime runtime{.user_data = descriptor, .userdata = &reads,
                           .read_specialization_memory = RejectSpecializationRead};
  FakeGuestMemory fake(RejectSpecializationRead, &reads);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, empty, populated;
  for (const uint32_t records : {0u, 1u, 0x20000000u, 0u}) {
    descriptor[2] = records;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0u &&
              specialization.buffers[0].scalar_empty == (records == 0u),
          "scalar emptiness read table contents or truncated its 64-bit byte size");
    if (records == 0u) {
      if (empty.buffers.empty()) empty = specialization;
      else Check(specialization == empty, "returning to an empty scalar table changed its specialization");
    }
    else if (records == 1u) populated = specialization;
    else Check(specialization == populated, "nonempty scalar size created a shader permutation");
    auto current = make_fixture();
    ApplyResourceSpecialization(current.program, specialization);
    for (const auto &[use, type] : current.uses) {
      const auto value = use->Arg(0).Resolve();
      Check(value.GetType() == type, "empty scalar-table folding changed the load result type");
      if (records != 0u) {
        Check(value.TryInstruction() != nullptr && BufferAccessOf(value.Instruction()->GetOpcode()) == BufferAccess::Read,
              "nonempty scalar table lost a runtime buffer load");
      } else if (value.IsImmediate()) {
        Check(value == Value(uint8_t{0}) || value == Value(uint16_t{0}) || value == Value(0u),
              "empty scalar table did not produce a typed zero");
      } else {
        for (size_t component = 0; component < value.Instruction()->NumArgs(); ++component)
          Check(value.Instruction()->Arg(component) == Value(0u),
                "empty scalar table retained a nonzero vector component");
      }
    }
    Check(current.nonnull->Arg(0).Resolve().TryInstruction() != nullptr,
          "a partly nonzero indirect descriptor was treated as unbound");
  }
  Check(empty != populated, "empty-to-populated scalar transition reused the folded specialization");
  plan.info.buffers[0].scalar = false;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            !specialization.buffers[0].scalar_empty,
        "vector-only resource acquired scalar-empty specialization");
  empty.buffers[0].indirect_root = 0u;
  empty.buffers.push_back(empty.buffers[0]);
  empty.buffers[1].scalar_empty = false;
  ApplyResourceSpecialization(fixture.program, empty);
  for (const auto &[use, type] : fixture.uses)
    Check(use->Arg(0).GetType() == type && use->Arg(0).Resolve().TryInstruction() != nullptr &&
              BufferAccessOf(use->Arg(0).ResolveInstruction()->GetOpcode()) == BufferAccess::Read,
          "empty root folded a scalar read with nonempty candidates");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = Extract(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

void TestImageMinimumLodSpecialization() {
  using namespace Libs::Graphics;
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto make_program = [](bool mixed) {
    auto program = MixedSamplerProgram();
    program.user_data_count = 24;
    auto &block = *program.blocks[0];
    for (uint32_t source = 0, reg = 0; source < program.descriptor_sources.size(); ++source) {
      for (uint32_t word = 0; word < program.descriptor_sources[source].dword_count; ++word, ++reg)
        program.descriptor_sources[source].dwords[word] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(reg))}));
    }
    program.info.samplers.resize(mixed ? 2 : 1);
    program.info.sampled_pairs = {{0, 0}, {1, 0}};
    if (mixed) program.info.sampled_pairs.push_back({0, 1});
    return program;
  };
  auto program = make_program(false);
  const auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 24> data{};
  for (uint32_t image = 0; image < 2; ++image) {
    data[image * 8] = 0x1000u + image * 0x100u;
    data[image * 8 + 1] = (static_cast<uint32_t>(Prospero::BufferFormat::k32Float) << 20u) | ((image == 0 ? 768u : 256u) << 8u);
    data[image * 8 + 3] = (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u) | (6u << 16u) | DstSel(4, 5, 6, 7);
  }
  data[17] = data[21] = 4095u << 12u;
  data[18] = data[22] = 2u << 26u;
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization native, shader;
  Check(MaterializeResources(plan, runtime, snapshot, native), "native image minimum LOD materialization failed");
  const auto samplers = snapshot.samplers;
  Check(MaterializeResources(plan, runtime, snapshot, shader, true) && shader != native &&
            shader.images[0].sample_min_lod == 768 && shader.images[1].sample_min_lod == 256 && snapshot.samplers == samplers,
        "distinct image shader clamps changed the shared sampler or lost specialization identity");
  const auto original = shader;
  data[1] = (data[1] & ~0xfff00u) | (384u << 8u);
  data[3] |= 1u << 12u;
  Check(MaterializeResources(plan, runtime, snapshot, shader, true) && shader != original && shader.images[0].sample_min_lod == 128,
        "base-relative fractional shader minimum did not change specialization");
  data[1] = (data[1] & ~0xfff00u) | (768u << 8u);
  data[3] &= ~(0xfu << 12u);
  Check(MaterializeResources(plan, runtime, snapshot, shader, true) && shader == original, "returning to the original minima lost its specialization");
  ApplyResourceSpecialization(program, shader);
  Check(program.info.samplers.size() == 1 && program.info.sampled_pairs[0].sampler == program.info.sampled_pairs[1].sampler &&
            program.info.images[0].sample_min_lod == 768 && program.info.images[1].sample_min_lod == 256,
        "applying historical shader minima duplicated a sampler or lost image constants");
  data[17] = 0;
  Check(!MaterializeResources(plan, runtime, snapshot, shader, true), "sampler maximum below image minimum was accepted");
  data[17] = 4095u << 12u;
  data[18] |= 1u << 20u;
  Check(!MaterializeResources(plan, runtime, snapshot, shader, true), "unequal mag/min filters were accepted");
  for (uint32_t filter : {2u, 3u}) {
    data[18] = (2u << 26u) | (filter << 20u) | (filter << 22u);
    Check(MaterializeResources(plan, runtime, snapshot, shader, true) && shader == original, "ratio-one anisotropic filter enum was rejected or changed image minima");
    data[16] = 1u << 9u;
    Check(!MaterializeResources(plan, runtime, snapshot, shader, true), "anisotropic filtering above ratio one was accepted");
    data[16] = 0;
  }
  data[18] = 2u << 26u;
  data[16] = 1u << 15u;
  Check(!MaterializeResources(plan, runtime, snapshot, shader, true), "unnormalized sampler coordinates were accepted");
  Check(MaterializeResources(plan, runtime, snapshot, native), "native minimum LOD inherited shader-only restrictions");
  data[16] = 0;
  auto dynamic_program = make_program(false);
  auto dynamic_plan = ExtractResourcePlan(dynamic_program);
  dynamic_plan.info.images[0].mip_mode = ImageMipMode::Dynamic;
  dynamic_plan.info.images[0].min_lod_sampler_exact = false;
  data[9] &= ~0xfff00u;
  Check(MaterializeResources(dynamic_plan, runtime, snapshot, shader, true) && shader.images[0].mip_count == 7 &&
            shader.images[0].sample_min_lod == 0 && shader.images[0].view_min_level == 0,
        "the existing per-mip shader path acquired a second historical minimum clamp");
  data[18] = 0;
  data[1] = (data[1] & ~0xfff00u) | (320u << 8u);
  data[9] = (data[9] & ~0xfff00u) | (448u << 8u);
  Check(MaterializeResources(plan, runtime, snapshot, shader, true) && shader.images[0].view_min_level == 1 &&
            shader.images[1].view_min_level == 1 && shader.images[0].sample_min_lod == 0 && shader.images[1].sample_min_lod == 0,
        "unmipmapped views did not floor their historical minimum level");
  auto mixed_program = make_program(true);
  const auto mixed_plan = ExtractResourcePlan(mixed_program);
  data[18] = 2u << 26u;
  data[22] = 0;
  data[1] = (data[1] & ~0xfff00u) | (768u << 8u);
  Check(MaterializeResources(mixed_plan, runtime, snapshot, shader, true) && shader.images[0].view_min_level == 3 && shader.images[0].sample_min_lod == 0,
        "mixed samplers did not share the historical integral view shift");
  ApplyResourceSpecialization(mixed_program, shader);
  Check(mixed_program.info.images[0].view_min_level == 3, "view minimum was not propagated to native IR");
  data[1] = (data[1] & ~0xfff00u) | (896u << 8u);
  Check(!MaterializeResources(mixed_plan, runtime, snapshot, shader, true), "mixed samplers accepted a fractional view shift");

  // E24/A170 uses the historical T# and S# of pixel shader 0xc924afb0821b68c8.
  const std::array<uint32_t, 24> e24 = {
      0x70d1e300u, 0xcb640000u, 0x01ffc1ffu, 0x909b0facu, 0x00000000u, 0x000000b0u, 0x0000002fu, 0x00000000u,
      0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
      0x00007800u, 0x00fff000u, 0x0af00000u, 0x00000000u};
  const auto materialize = [&](const std::array<uint32_t, 24> &words, ResourceSpecialization &specialization, ShaderType stage = ShaderType::Pixel, bool implicit = true, bool dynamic_mips = false, bool mixed = false) {
    auto current = make_program(mixed);
    current.stage = stage;
    current.info.images[0].implicit_only = implicit;
    if (dynamic_mips) current.info.images[0].mip_mode = ImageMipMode::Dynamic;
    const auto current_plan = ExtractResourcePlan(current);
    ResourceSnapshot current_snapshot;
    return MaterializeResources(current_plan, {.user_data = words}, current_snapshot, specialization, true);
  };
  Check(materialize(e24, shader) && shader.images[0].sample_min_lod == 1024u && shader.images[0].view_min_level == 0u,
        "E24 implicit pixel sample did not keep MIN_LOD 4.0 as its MinLod operand");
  // Integral minimum LOD with a negative anisotropic sampler bias.
  const std::array<uint32_t, 24> biased_image = {
      0x71b12c00u, 0xcb610000u, 0x03ffc3ffu, 0x909c03acu, 0x00000000u, 0x000000c0u, 0x00000000u, 0x00000000u,
      0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
      0x00007800u, 0x00fff000u, 0x0af03fccu, 0x00000000u};
  const auto materialize_biased_image = [&](const std::array<uint32_t, 24> &input, ResourceSpecialization &specialization, ShaderType stage = ShaderType::Pixel, bool implicit = true, bool emulate = true, bool dynamic = false, bool mixed = false) {
    auto current = make_program(mixed);
    current.shader_hash = 0xcb9ba1c87080e9d9ull;
    current.stage = stage;
    const auto captured = current.info.images[0];
    current.info.images[0] = current.info.images[1];
    current.info.images.resize(7, current.info.images[1]);
    current.info.images[6] = captured;
    current.info.images[6].first_use_pc = 0xaf4u;
    current.info.images[6].implicit_only = implicit;
    current.info.images[6].mip_mode = dynamic ? ImageMipMode::Dynamic : ImageMipMode::None;
    current.info.sampled_pairs = {{6, 0, 0xaf4u}};
    if (mixed) current.info.sampled_pairs.push_back({6, 1, 0xaf4u});
    const auto current_plan = ExtractResourcePlan(current);
    ResourceSnapshot current_snapshot;
    const bool ok = MaterializeResources(current_plan, {.user_data = input}, current_snapshot, specialization, emulate);
    if (ok) {
      Check(std::equal(input.begin(), input.begin() + 8, current_snapshot.images[6].dwords.begin()) &&
                std::equal(input.begin() + 16, input.begin() + 20, current_snapshot.samplers[0].dwords.begin()),
            "biased_image minimum LOD changed the captured image or sampler descriptor");
      ApplyResourceSpecialization(current, specialization);
      Check(current.info.images[6].view_min_level == specialization.images[6].view_min_level &&
                current.info.images[6].sample_min_lod == specialization.images[6].sample_min_lod &&
                current.info.samplers[0].snapshot_index == 0u && current.info.sampled_pairs[0].image == 6u,
            "biased_image view shift or sampler identity did not reach native IR");
    }
    return ok;
  };
  Check(materialize_biased_image(biased_image, shader) && shader.images[6].view_min_level == 1u && shader.images[6].sample_min_lod == 0u && shader.images[6].mip_count == 1u,
        "captured biased anisotropic biased_image sampler did not select the exact level-1 view");
  const auto shifted_biased_image = shader;
  Check(materialize_biased_image(biased_image, native, ShaderType::Pixel, true, false) && native.images[6].view_min_level == 0u && native.images[6].sample_min_lod == 0u && native != shifted_biased_image,
        "native image minimum LOD inherited the emulated shifted view");
  Check(materialize_biased_image(biased_image, shader, ShaderType::Pixel, true, true, true) && shader.images[6].mip_count == 13u && shader.images[6].view_min_level == 0u && shader.images[6].sample_min_lod == 0u,
        "biased_image dynamic per-mip path acquired a second minimum LOD emulation");
  const auto biased_image_refused = [&](const std::array<uint32_t, 24> &input, const char *message, ShaderType stage = ShaderType::Pixel, bool implicit = true) {
    ResourceSpecialization rejected;
    Check(!materialize_biased_image(input, rejected, stage, implicit), message);
  };
  biased_image_refused(biased_image, "biased biased_image sampler admitted for an explicit or mixed image use", ShaderType::Pixel, false);
  biased_image_refused(biased_image, "biased biased_image sampler admitted in a vertex shader", ShaderType::Vertex);
  biased_image_refused(biased_image, "biased biased_image sampler admitted in a compute shader", ShaderType::Compute);
  auto biased_image_words = biased_image;
  biased_image_words[18] |= 1u << 14u;
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with a secondary bias");
  biased_image_words = biased_image;
  biased_image_words[18] = (biased_image_words[18] & ~0x3fffu) | 513u;
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted beyond the guaranteed native bias range");
  biased_image_words = biased_image;
  biased_image_words[1] = (biased_image_words[1] & ~0xfff00u) | (384u << 8u);
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with a fractional minimum LOD");
  biased_image_words = biased_image;
  biased_image_words[1] &= ~(1u << 30u);
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with a width that does not halve exactly");
  biased_image_words = biased_image;
  biased_image_words[2] = (biased_image_words[2] & ~(0x3fffu << 14u)) | (4094u << 14u);
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with a height that does not halve exactly");
  biased_image_words = biased_image;
  biased_image_words[17] = (3071u << 12u);
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with MAX_LOD below the original last level");
  biased_image_words = biased_image;
  biased_image_words[17] |= 1u;
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with a nonzero sampler minimum");
  biased_image_words = biased_image;
  biased_image_words[16] |= 1u << 15u;
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with unnormalized coordinates");
  biased_image_words = biased_image;
  biased_image_words[18] = (biased_image_words[18] & ~(3u << 20u)) | (2u << 20u);
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with unequal mag/min filters");
  biased_image_words = biased_image;
  biased_image_words[18] |= 1u << 26u;
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted with an unsupported mip filter");
  biased_image_words = biased_image;
  biased_image_words[18] &= ~(3u << 26u);
  biased_image_refused(biased_image_words, "biased biased_image sampler admitted without mipmapping");
  biased_image_words = biased_image;
  std::copy_n(biased_image.begin() + 16, 4, biased_image_words.begin() + 20);
  biased_image_words[22] &= ~(3u << 26u);
  Check(!materialize_biased_image(biased_image_words, shader, ShaderType::Pixel, true, true, false, true),
        "biased biased_image image admitted an unmipmapped companion sampler");
  Check(materialize_biased_image(biased_image, shader) && shader == shifted_biased_image,
        "captured biased_image view specialization did not survive rematerialization after refusals");
  const auto refused = [&](const std::array<uint32_t, 24> &words, const char *message, ShaderType stage = ShaderType::Pixel, bool implicit = true) {
    ResourceSpecialization specialization;
    Check(!materialize(words, specialization, stage, implicit), message);
  };
  refused(e24, "E24 sampler admitted for a non-implicit use", ShaderType::Pixel, false);
  refused(e24, "E24 sampler admitted in a vertex shader", ShaderType::Vertex);
  refused(e24, "E24 sampler admitted in a compute shader", ShaderType::Compute);
  auto words = e24;
  words[18] |= 0x100u;
  Check(materialize(words, shader) && shader.images[0].view_min_level == 4u && shader.images[0].sample_min_lod == 0u,
        "E24 biased implicit pixel sample did not shift its view to level 4");
  words = e24;
  words[18] |= 1u << 14u;
  refused(words, "E24 sampler admitted with a secondary LOD bias");
  words = e24;
  words[18] = (words[18] & ~(3u << 20u)) | (2u << 20u);
  refused(words, "E24 sampler admitted with mag != min filters");
  words = e24;
  words[17] = (words[17] & ~(0xfffu << 12u)) | (1023u << 12u);
  refused(words, "E24 sampler admitted with MAX_LOD below MIN_LOD 4.0");
  words = e24;
  words[16] |= 1u << 15u;
  refused(words, "E24 sampler admitted with unnormalized coordinates");
  words = e24;
  words[18] |= 1u << 26u;
  refused(words, "E24 sampler admitted with an unsupported mip filter");
  words = e24;
  words[17] = (1024u << 12u) | 1025u;
  refused(words, "E24 sampler admitted with inverted sampler clamps");
  words = e24;
  words[18] &= ~(3u << 26u);
  refused(words, "E24 anisotropic sampler admitted without mipmapping");
  Check(!materialize(e24, shader, ShaderType::Pixel, true, false, true), "E24 anisotropic sampler admitted with an unmipmapped companion");
  Check(materialize(e24, shader, ShaderType::Pixel, true, true) && shader.images[0].sample_min_lod == 0u && shader.images[0].view_min_level == 0u,
        "E24 dynamic mip path acquired the implicit pixel exception");
}

void TestSamplerVariantRemaps(bool compact) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  using D = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  Program program;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  const uint32_t root = compact ? 1u : 0u, direct = root + 2u;
  program.info.images.resize(direct + 1u);
  ResourceSpecialization specialization;
  specialization.images.resize(program.info.images.size());
  if (compact) specialization.images[0].fmask = true;
  for (uint32_t image = root; image <= direct; ++image) {
    program.info.images[image].resource_class = ImageResourceClass::Sampled;
    auto &type = specialization.images[image];
    type.numeric_class = image == root + 1u ? N::Uint : N::Float;
    type.dimension = D::Dim2D;
    if (image != direct) type.indirect_root = root;
    type.indirect_mapping_offset = 19;
    type.indirect_search_iterations = 2;
  }
  program.info.samplers = {{.source = 20}, {.source = 21}};
  specialization.samplers.resize(2);
  for (auto &sampler : specialization.samplers) {
    sampler.indirect_root = 0;
    sampler.indirect_mapping_offset = 37;
    sampler.indirect_search_iterations = 3;
  }
  program.info.sampled_pairs = {{root, 0}, {direct, 0}};
  std::vector<Inst *> handles;
  for (const auto image : {root, direct}) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetImageResource, {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u)});
    handle.SetFlags(image);
    handles.push_back(&handle);
    const auto memory_index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back({.kind = ResourceKind::Image, .resource = image, .sampler = 0});
    auto &sampler = block.AppendNewInst(ValueOpcode::GetSamplerResource, {Value(0u), Value(0u), Value(0u), Value(0u)});
    auto &address = block.AppendNewInst(ValueOpcode::MakeImageAddress, {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u)});
    auto &sample = block.AppendNewInst(ValueOpcode::ImageSampleRaw, {Value(&handle), Value(&sampler), Value(&address), Value(true)});
    sample.SetFlags(MemoryFlags{.index = memory_index});
  }
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.images.size() == 3 && program.info.samplers.size() == 4u,
        "typed variants exceeded the required structural set");
  for (uint32_t pair_index = 0; pair_index < 2; ++pair_index) {
    const auto &pair = program.info.sampled_pairs[pair_index];
    const auto &memory = program.memory_info[pair_index];
    Check(pair.image == (pair_index == 0 ? 0u : 2u) && handles[pair_index]->Flags<uint32_t>() == pair.image &&
              memory.resource == pair.image && memory.sampler == pair.sampler,
          "instruction and sampled-pair remaps disagree after image compaction");
    const auto &sampler = program.info.samplers[pair.sampler];
    Check(sampler.snapshot_index == 0 &&
              sampler.indirect_root != SamplerResource::NoIndirectSampler && sampler.indirect_resources.size() == 2 &&
              sampler.indirect_mapping_offset == 37 && sampler.indirect_search_iterations == 3,
          "indirect sampler root lost selector provenance or candidates");
  }
  for (const auto &sampler : program.info.samplers) {
    Check(sampler.source == 20u + sampler.snapshot_index && sampler.force_point_filtering == sampler.integer_border,
          "typed sampler cloning lost source provenance");
    for (const auto candidate : sampler.indirect_resources) {
      const auto &child = program.info.samplers[candidate];
      Check(child.integer_border == sampler.integer_border &&
                child.indirect_root != SamplerResource::NoIndirectSampler &&
                child.indirect_mapping_offset == sampler.indirect_mapping_offset && child.indirect_search_iterations == sampler.indirect_search_iterations,
            "typed indirect candidate lost selector provenance");
    }
  }
  Check(program.info.images[0].indirect_resources == std::vector<uint32_t>({0, 1}) &&
            program.info.images[0].indirect_mapping_offset == 19,
        "image compaction changed candidate ordinals or SRT mapping");
}

void TestSharedSamplerCandidateProvenance() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.resource_tracking_complete = true;
  program.info.images.resize(2);
  program.info.samplers = {{.source = 20}, {.source = 21}, {.source = 22}};
  program.info.samplers[1].indirect_resources = {1, 2};
  program.info.sampled_pairs = {{0, 0}, {1, 1}};
  ResourceSpecialization specialization;
  specialization.images.resize(2);
  specialization.samplers.resize(3);
  for (uint32_t image = 0; image < 2; ++image) {
    program.info.images[image].resource_class = ImageResourceClass::Sampled;
    specialization.images[image].dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D;
    specialization.images[image].numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float;
  }
  for (const auto source : {0u, 2u}) {
    specialization.samplers[source].indirect_root = 0;
    specialization.samplers[source].indirect_mapping_offset = 37;
    specialization.samplers[source].indirect_search_iterations = 3;
  }
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.samplers.size() == 3, "shared candidate added an unnecessary sampler variant");
  for (const auto &pair : program.info.sampled_pairs) {
    const auto &root = program.info.samplers[pair.sampler];
    Check(root.indirect_resources.size() == 2, "shared candidate changed selector ordinals");
    const auto &candidate = program.info.samplers[root.indirect_resources[1]];
    Check(candidate.snapshot_index == 2 && candidate.source == 22 &&
              candidate.indirect_root != SamplerResource::NoIndirectSampler &&
              candidate.indirect_mapping_offset == 37 && candidate.indirect_search_iterations == 3,
          "shared candidate lost retained selector provenance");
  }
}

void TestSignedStorageSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using F = Libs::Graphics::Prospero::BufferFormat;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 1u;
  auto &block = AddValueBlock(program);
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  constexpr auto swizzle = Libs::Graphics::DstSel(5, 4, 1, 0);
  source.dwords[3] = Value(swizzle |
      (uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
  program.descriptor_sources.push_back(source);
  program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D, .written = true});
  auto plan = Extract(program);
  std::array<uint32_t, 1> data{};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, signed16;
  for (const auto format : {F::k16SInt, F::k32SInt, F::k16SInt}) {
    data[0] = uint32_t(format) << 20u;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "write-only signed storage image failed materialization");
    const auto conversion = format == F::k16SInt ? format : F::kInvalid;
    const auto view = format == F::k16SInt ? F::k16UInt : F::k32UInt;
    Check(specialization.images[0].numeric_class == N::Uint &&
              specialization.images[0].conversion_format == conversion &&
              specialization.images[0].shader_swizzle == swizzle &&
              snapshot.images[0].dwords[1] == data[0] &&
              RawSignedStorageFormat(plan.info.images[0], format) == view,
          "signed storage specialization lost guest format, conversion, swizzle or raw view width");
    if (format == F::k16SInt) {
      if (signed16.images.empty()) signed16 = specialization;
      else Check(specialization == signed16, "signed storage format refresh retained a stale specialization");
    } else {
      Check(specialization != signed16, "signed 16/32-bit stores shared incompatible conversion state");
    }
    ApplyResourceSpecialization(program, specialization);
    const auto binding = DescriptorBindingForImage(program.info.images[0]);
    Check(binding && ImageBindingResourceClass(*binding) == ImageResourceClass::Storage,
          "signed storage conversion did not reuse the unsigned storage binding");
  }
  for (const auto format : {F::k8SInt, F::k16SInt, F::k32SInt}) {
    data[0] = uint32_t(format) << 20u;
    auto &image = plan.info.images[0];
    image.read = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed raw storage view accepted a non-atomic read/write image");
    image.atomic = true;
    Check((RawSignedStorageFormat(image, format) != F::kInvalid) == (format == F::k32SInt) &&
              MaterializeResources(plan, runtime, snapshot, specialization) == (format == F::k32SInt),
          "signed storage conversion changed atomic format eligibility");
    image.atomic64 = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed 16/32-bit image acquired a 64-bit atomic raw view");
    image.resource_class = ImageResourceClass::Sampled;
    image.written = image.atomic = image.atomic64 = false;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == N::Sint &&
              specialization.images[0].conversion_format == F::kInvalid &&
              Libs::Graphics::Prospero::NumBytesPerElement(format) ==
                  (format == F::k8SInt ? 1u : format == F::k16SInt ? 2u : 4u) &&
              RawSignedStorageFormat(image, format) == F::kInvalid,
          "storage conversion changed ordinary signed image reads");
    image.resource_class = ImageResourceClass::Storage;
    image.written = true;
    image.read = false;
  }
}

void TestNullDescriptorFallbackTransitions() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using namespace Libs::Graphics;
  Program program;
  program.stage = ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 2u;
  auto &block = AddValueBlock(program);
  DescriptorSource image;
  image.dword_count = 8u;
  image.dwords.fill(Value(0u));
  image.dwords[0] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  image.dwords[1] = Value(uint32_t(Prospero::BufferFormat::k32Float) << 20u);
  image.dwords[3] = Value(DstSel(4, 5, 6, 7) | (uint32_t(Prospero::ImageType::kColor2D) << 28u));
  program.descriptor_sources.push_back(image);
  DescriptorSource buffer;
  buffer.dword_count = 4u;
  buffer.dwords.fill(Value(0u));
  buffer.dwords[0] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(1))}));
  buffer.dwords[2] = Value(16u);
  program.descriptor_sources.push_back(buffer);
  program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Sampled, .read = true});
  program.info.buffers.push_back({.source = 1u, .read = true, .written = true});
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> data{};
  SrtRuntime runtime{.user_data = data, .null_descriptor_supported = false};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, native;
  for (const uint32_t address : {0u, 1u, 0u}) {
    data.fill(address);
    Check(MaterializeResources(plan, runtime, snapshot, specialization), "null fallback refresh failed");
    Check(specialization.images[0].null_fallback == (address == 0u) && specialization.buffers[0].null_fallback == (address == 0u), "null/real transition retained stale fallback state");
    runtime.null_descriptor_supported = true;
    Check(MaterializeResources(plan, runtime, snapshot, native) && !native.images[0].null_fallback && !native.buffers[0].null_fallback, "native null descriptors entered fallback");
    Check((specialization == native) == (address != 0u), "null fallback state is missing from the permutation key");
    runtime.null_descriptor_supported = false;
  }
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.images[0].null_fallback && program.info.buffers[0].null_fallback, "fallback state was not propagated to native IR");
}

// One descriptor serves a store, an atomic that a user-data condition guards, and an image no
// block lists. Unreached, the atomic reads a zero descriptor instead of the RGBA8 format the
// others use; reached, it rejects that format.
void TestUnreachedImageIgnoresSharedDescriptor(bool atomic64) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  using Libs::Graphics::Prospero::BufferFormat;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 2u;
  auto &entry = AddValueBlock(program);
  auto &guarded = AddValueBlock(program);
  auto &done = AddValueBlock(program);
  const auto user = [&](uint32_t index) {
    return Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(index))}));
  };
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = user(0);
  source.dwords[3] = Value(uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  program.descriptor_sources.push_back(source);
  const auto dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D;
  for (const bool atomic : {false, true, false})
    program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
        .dimension = dimension, .written = true, .atomic = atomic, .atomic64 = atomic && atomic64});
  for (uint32_t image = 0; image < 2u; ++image)
    program.memory_info.push_back({.kind = ResourceKind::Image, .resource = image});
  entry.AppendNewInst(ValueOpcode::ImageWrite, {Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 0});
  guarded.AppendNewInst(atomic64 ? ValueOpcode::ImageAtomicUMax64 : ValueOpcode::ImageAtomicUMax32,
                        {Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  entry.condition = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {user(1), Value(0u)}));
  entry.terminator.kind = TerminatorKind::ConditionalBranch;
  entry.terminator.true_block = &guarded;
  entry.terminator.false_block = &done;
  guarded.terminator.kind = TerminatorKind::Branch;
  guarded.terminator.true_block = &done;
  done.terminator.kind = TerminatorKind::Return;
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> data{uint32_t(BufferFormat::k8_8_8_8UNorm) << 20u, 0u};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.images[0].dwords[1] == data[0] && snapshot.images[2].dwords[1] == data[0] &&
            snapshot.images[1] == DescriptorValue{.dword_count = 8u},
        "unreached atomic image kept the shared descriptor or an unlisted image lost it");
  data[1] = 1u;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization),
        "reached atomic image accepted a format its width rejects");
}

void TestIndirectImageSamplerClasses(bool mixed, bool min_lod_exact) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::TextureNumericClass;
  auto program = MixedSamplerProgram();
  auto &first = program.descriptor_sources[0];
  first.dwords[0] = Value(0x100u);
  first.dwords[1] = Value(static_cast<uint32_t>(BufferFormat::k8UNorm) << 20u);
  first.dwords[3] = program.descriptor_sources[1].dwords[3];
  program.descriptor_sources[1].dwords[1] = Value(static_cast<uint32_t>(
      mixed ? BufferFormat::k8UInt : BufferFormat::k8UNorm) << 20u);
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 0};
  program.info.images.resize(1);
  program.info.images[0].source = static_cast<uint32_t>(program.descriptor_sources.size());
  program.info.images[0].min_lod_sampler_exact = min_lod_exact;
  program.descriptor_sources.push_back(root);
  program.info.samplers.resize(1);
  program.info.sampled_pairs.resize(1);
  auto plan = Extract(program);
  std::array<uint32_t, 1> user_data{0x11111111u};
  uint32_t reads = 0;
  const SrtRuntime runtime{.user_data = user_data, .userdata = &reads,
                           .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0 &&
            snapshot.images.size() == 2 && snapshot.images[0].dwords[0] == 0x100u &&
            snapshot.images[1].dwords[0] == 1u &&
            specialization.images[0].numeric_class == TextureNumericClass::Float &&
            specialization.images[1].numeric_class ==
                (mixed ? TextureNumericClass::Uint : TextureNumericClass::Float),
        "indirect image materialization rejected or changed a typed candidate");
  const auto offset = specialization.images[0].indirect_mapping_offset;
  Check(snapshot.flattened_srt[offset] == 3u &&
            snapshot.flattened_srt[offset + 2u] == 0u &&
            snapshot.flattened_srt[offset + 4u] == 1u &&
            snapshot.flattened_srt[offset + 6u] == 0u,
        "mixed image classes changed selector ordinals or duplicated candidates");
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.images[0].indirect_resources == std::vector<uint32_t>({0, 1}) &&
            snapshot.samplers.size() == 1 && program.info.samplers.size() == (mixed ? 2u : 1u),
        "indirect images lost candidates or created unnecessary sampler variants");
  Check(plan.info.images[0].min_lod_sampler_exact == min_lod_exact &&
            std::ranges::all_of(program.info.images, [&](const auto &image) { return image.min_lod_sampler_exact == min_lod_exact; }),
        "minLod eligibility changed during extraction, candidate expansion or duplicate compaction");
  user_data[0] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed image sampler refresh failed");
  Check(std::ranges::all_of(program.info.images, [&](const auto &image) { return image.min_lod_sampler_exact == min_lod_exact; }),
        "descriptor refresh lost the candidate minLod eligibility proof");
  for (uint32_t index = 0; index < program.info.samplers.size(); ++index) {
    const auto &sampler = program.info.samplers[index];
    Check(sampler.snapshot_index == 0u && sampler.source == plan.info.samplers[0].source &&
              sampler.force_point_filtering == (index != 0u) &&
              sampler.integer_border == (index != 0u) &&
              snapshot.samplers[sampler.snapshot_index].dwords[0] == user_data[0],
          "typed sampler variants lost filtering, border class, or shared live descriptor");
  }
}

void TestShaderCacheRecords() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Equivalent selector domains share one key group, which a load rebuilds.
  Program selectors;
  AddValueBlock(selectors);
  for (const uint32_t stride : {4u, 8u, 4u}) {
    DescriptorSource source;
    source.indirect_descriptor = DescriptorSource::IndirectDescriptor{
        .selector = DescriptorSource::IndirectDescriptor::SelectorRead{.stride = stride}};
    selectors.descriptor_sources.push_back(source);
  }
  selectors.descriptor_sources.emplace_back();
  const auto plan = Extract(selectors);
  Check(plan.selector_groups == std::vector<uint32_t>{0u, 1u, 0u, UINT32_MAX} &&
            plan.selector_keys.size() == 2u,
        "loaded plan lost its selector key groups");

  const std::array values{Value(), Value(ScalarReg(3)), Value(VectorReg(5)), Value(true),
                          Value(uint8_t{7}), Value(uint16_t{9}), Value(11u), Value(uint64_t{1} << 40u),
                          Value::F16(0x3c00u), Value::F32(-2.5f)};
  Writer writer;
  Io(writer, values);
  Reader reader{writer.bytes};
  std::array<Value, values.size()> loaded;
  Io(reader, loaded);
  Check(reader.ok && reader.bytes.empty() && loaded == values,
        "immediate values changed through the shader cache");
  // Damaged records: a missing instruction, a type that is no value type, an oversized count and
  // an unknown opcode.
  for (const uint8_t type : {1u, 3u}) {
    const std::array<uint8_t, 12> bytes{type};
    Reader damaged{bytes};
    Value value;
    Io(damaged, value);
    Check(!damaged.ok, "damaged value was accepted");
  }
  ResourcePlan oversized, unknown;
  Check(!LoadResourcePlan(std::array<uint8_t, 4>{0xffu, 0xffu, 0xffu, 0xffu}, oversized) &&
            !LoadResourcePlan(std::array<uint8_t, 14>{1u, 0u, 0u, 0u, 0xffu, 0xffu}, unknown),
        "damaged resource plan was accepted");
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  TestWalkerLoadsBypassSpecializationReader();
  TestNativeNestedSrtRefresh();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestSrtAliasesShareReads();
  TestProtectedPageLoadsCallReader();
  TestUniformVectorDescriptorRead();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  for (const bool memory_condition : {false, true})
    TestWrittenDescriptorPredicateReads(memory_condition);
  TestFailedMaterializationRejectsStage();
  TestSharedControlPredicates();
  TestNestedExecPredicatesReuseReads();
  TestFiniteImageRefreshReusesScalarReads();
  for (const bool independent_mask : {false, true})
    TestEmptyTraversalSkipsUnmappedDescriptor(independent_mask);
  TestEmptyTraversalSkipsUnmappedDescriptor(false, true);
  TestCyclicControlPredicatesRemainUnknown();
  TestBackEdgeReachesPhiPredecessor();
  TestEmptyScalarBufferSpecialization();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestImageMinimumLodSpecialization();
  TestSharedSamplerCandidateProvenance();
  for (const bool compact : {false, true}) TestSamplerVariantRemaps(compact);
  TestSignedStorageSpecialization();
  TestNullDescriptorFallbackTransitions();
  for (const bool atomic64 : {false, true}) TestUnreachedImageIgnoresSharedDescriptor(atomic64);
  for (const bool mixed : {false, true}) {
    for (const bool exact : {false, true}) TestIndirectImageSamplerClasses(mixed, exact);
  }
  TestShaderCacheRecords();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
