#include "libs/libJpegDec.cpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Loader {
void SymbolDatabase::Add(const SymbolResolve&, uint64_t, const std::string&) {}
namespace Timer {
double GetTimeMs() {
	return 0.0;
}
} // namespace Timer
} // namespace Loader

namespace {
using namespace Libs::JpegDec;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "JpegDecTests:%d: %s\n", __LINE__, #condition);                   \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

constexpr uint32_t WIDTH  = 16;
constexpr uint32_t HEIGHT = 8;

// Left half red, right half blue: large flat areas survive JPEG quantization.
std::vector<uint8_t> MakeJpeg(int components = 3) {
	std::vector<uint8_t> pixels(static_cast<size_t>(WIDTH) * HEIGHT * components);
	for (uint32_t y = 0; y < HEIGHT; y++) {
		for (uint32_t x = 0; x < WIDTH; x++) {
			auto* p = &pixels[(static_cast<size_t>(y) * WIDTH + x) * components];
			if (components == 1) {
				p[0] = (x < WIDTH / 2 ? 40 : 200);
			} else {
				p[0] = (x < WIDTH / 2 ? 255 : 0);
				p[1] = 0;
				p[2] = (x < WIDTH / 2 ? 0 : 255);
			}
		}
	}

	std::vector<uint8_t> jpeg;
	auto                 write = [](void* context, void* data, int size) {
		auto* out = static_cast<std::vector<uint8_t>*>(context);
		out->insert(out->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
	};
	CHECK(stbi_write_jpg_to_func(write, &jpeg, WIDTH, HEIGHT, components, pixels.data(), 100) != 0);
	return jpeg;
}

bool Near(uint8_t value, uint8_t expected) {
	return (value > expected ? value - expected : expected - value) <= 24;
}

void TestParseHeader() {
	const auto jpeg = MakeJpeg(3);

	JpegDecImageInfo        info {};
	const JpegDecParseParam parse {jpeg.data(), static_cast<uint32_t>(jpeg.size()), 0x10000};
	CHECK(JpegDecParseHeader(&parse, &info) == 0);
	CHECK(info.image_width == WIDTH);
	CHECK(info.image_height == HEIGHT);
	CHECK(info.color_space == JPEG_DEC_COLOR_SPACE_YCBCR);
	CHECK(info.image_format == JPEG_DEC_IMAGE_FORMAT_BASELINE);
	CHECK(info.coef_mem_size == 0);
	CHECK(info.suggested_max_image_width == WIDTH);

	// Hand-made headers (the stb writer always emits baseline YCbCr): a progressive grayscale
	// frame preceded by an APP0 segment and fill bytes.
	const uint8_t gray[] = {0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x00, 0x00, 0xff, 0xff, 0xc2, 0x00,
	                        0x0b, 0x08, 0x00, 0x30, 0x00, 0x40, 0x01, 0x01, 0x11, 0x00, 0xff, 0xd9};
	const JpegDecParseParam gray_parse {gray, sizeof(gray), 0};
	CHECK(JpegDecParseHeader(&gray_parse, &info) == 0);
	CHECK(info.image_width == 0x40 && info.image_height == 0x30);
	CHECK(info.color_space == JPEG_DEC_COLOR_SPACE_GRAYSCALE);
	CHECK(info.image_format == JPEG_DEC_IMAGE_FORMAT_PROGRESSIVE);

	// A scan before any frame header and a segment running past the buffer are rejected.
	const uint8_t           no_frame[] = {0xff, 0xd8, 0xff, 0xda, 0x00, 0x02, 0xff, 0xd9};
	const JpegDecParseParam no_frame_parse {no_frame, sizeof(no_frame), 0};
	CHECK(JpegDecParseHeader(&no_frame_parse, &info) == JPEG_DEC_ERROR_INVALID_DATA);
	const JpegDecParseParam cut_frame {gray, 16, 0};
	CHECK(JpegDecParseHeader(&cut_frame, &info) == JPEG_DEC_ERROR_INVALID_DATA);

	CHECK(JpegDecParseHeader(nullptr, &info) == JPEG_DEC_ERROR_INVALID_ADDR);
	CHECK(JpegDecParseHeader(&parse, nullptr) == JPEG_DEC_ERROR_INVALID_ADDR);

	const JpegDecParseParam truncated {jpeg.data(), 20, 0};
	CHECK(JpegDecParseHeader(&truncated, &info) == JPEG_DEC_ERROR_INVALID_DATA);

	const uint8_t           png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
	const JpegDecParseParam not_jpeg {png, sizeof(png), 0};
	CHECK(JpegDecParseHeader(&not_jpeg, &info) == JPEG_DEC_ERROR_INVALID_DATA);
}

void TestDecode(uint16_t pixel_format) {
	const auto jpeg = MakeJpeg(3);

	const JpegDecCreateParam create {sizeof(JpegDecCreateParam), 0, WIDTH};
	const int32_t            memory_size = JpegDecQueryMemorySize(&create);
	CHECK(memory_size >= static_cast<int32_t>(sizeof(JpegDecContext)));

	std::vector<uint8_t> memory(memory_size);
	void*                handle = nullptr;
	CHECK(JpegDecCreate(&create, memory.data(), static_cast<uint32_t>(memory.size()), &handle) ==
	      0);
	CHECK(handle != nullptr);

	constexpr uint32_t   pitch = WIDTH * 4 + 16;
	std::vector<uint8_t> image(static_cast<size_t>(pitch) * HEIGHT, 0xcd);

	JpegDecDecodeParam decode {};
	decode.jpeg_mem_addr  = jpeg.data();
	decode.image_mem_addr = image.data();
	decode.jpeg_mem_size  = static_cast<uint32_t>(jpeg.size());
	decode.image_mem_size = static_cast<uint32_t>(image.size());
	decode.decode_option  = 0x10000;
	decode.pixel_format   = pixel_format;
	decode.alpha_value    = 0xff;
	decode.image_pitch    = pitch;

	JpegDecImageInfo info {};
	CHECK(JpegDecDecode(handle, &decode, &info) == static_cast<int32_t>((WIDTH << 16u) | HEIGHT));
	CHECK(info.image_width == WIDTH && info.image_height == HEIGHT);
	CHECK(JpegDecDecode(handle, &decode, nullptr) >= 0);

	const bool bgra = (pixel_format == JPEG_DEC_PIXEL_FORMAT_B8G8R8A8);
	for (uint32_t y = 0; y < HEIGHT; y++) {
		const uint8_t* row = image.data() + static_cast<size_t>(y) * pitch;
		// Away from the color edge, where chroma subsampling blends the two halves.
		const uint8_t* left  = row + 1 * 4;
		const uint8_t* right = row + (WIDTH - 2) * 4;
		CHECK(Near(left[bgra ? 2 : 0], 255) && Near(left[1], 0) && Near(left[bgra ? 0 : 2], 0));
		CHECK(Near(right[bgra ? 2 : 0], 0) && Near(right[1], 0) && Near(right[bgra ? 0 : 2], 255));
		CHECK(left[3] == 0xff && right[3] == 0xff);
		// Padding past the row is left untouched.
		CHECK(row[WIDTH * 4] == 0xcd && row[pitch - 1] == 0xcd);
	}

	// Output buffer too small for the image at this pitch.
	decode.image_mem_size = pitch * (HEIGHT - 1);
	CHECK(JpegDecDecode(handle, &decode, nullptr) == JPEG_DEC_ERROR_INVALID_SIZE);
	decode.image_mem_size = static_cast<uint32_t>(image.size());

	decode.image_pitch = WIDTH * 4 - 4;
	CHECK(JpegDecDecode(handle, &decode, nullptr) == JPEG_DEC_ERROR_INVALID_SIZE);
	decode.image_pitch = pitch;

	decode.pixel_format = 7;
	CHECK(JpegDecDecode(handle, &decode, nullptr) == JPEG_DEC_ERROR_INVALID_PARAM);
	decode.pixel_format = pixel_format;

	// Valid header, corrupted entropy-coded data.
	auto broken = jpeg;
	broken.resize(broken.size() / 2);
	decode.jpeg_mem_addr        = broken.data();
	decode.jpeg_mem_size        = static_cast<uint32_t>(broken.size());
	const int32_t broken_result = JpegDecDecode(handle, &decode, nullptr);
	CHECK(broken_result == JPEG_DEC_ERROR_DECODE_ERROR || broken_result >= 0);

	CHECK(JpegDecDelete(handle) == 0);
	CHECK(JpegDecDelete(handle) == JPEG_DEC_ERROR_INVALID_HANDLE);
	CHECK(JpegDecDecode(handle, &decode, nullptr) == JPEG_DEC_ERROR_INVALID_HANDLE);
}

void TestCreateErrors() {
	JpegDecCreateParam create {sizeof(JpegDecCreateParam), 0, WIDTH};
	JpegDecContext     context {};
	void*              handle = nullptr;

	CHECK(JpegDecQueryMemorySize(nullptr) == JPEG_DEC_ERROR_INVALID_ADDR);
	CHECK(JpegDecCreate(&create, nullptr, sizeof(context), &handle) == JPEG_DEC_ERROR_INVALID_ADDR);
	CHECK(JpegDecCreate(&create, &context, sizeof(context) - 1, &handle) ==
	      JPEG_DEC_ERROR_INVALID_WORK_MEMORY);

	create.this_size = 8;
	CHECK(JpegDecQueryMemorySize(&create) == JPEG_DEC_ERROR_INVALID_SIZE);
	CHECK(JpegDecCreate(&create, &context, sizeof(context), &handle) ==
	      JPEG_DEC_ERROR_INVALID_SIZE);

	CHECK(JpegDecDelete(nullptr) == JPEG_DEC_ERROR_INVALID_HANDLE);
	CHECK(JpegDecDecode(nullptr, nullptr, nullptr) == JPEG_DEC_ERROR_INVALID_HANDLE);
}

} // namespace

int main() {
	TestParseHeader();
	TestDecode(JPEG_DEC_PIXEL_FORMAT_R8G8B8A8);
	TestDecode(JPEG_DEC_PIXEL_FORMAT_B8G8R8A8);
	TestCreateErrors();
	std::printf("JpegDecTests: OK\n");
	return 0;
}
