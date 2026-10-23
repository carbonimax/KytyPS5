#include "common/abi.h"
#include "common/logging/log.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"
#include "stb_image.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

// JPEG decoder library. The structure layouts below are inferred from how titles fill and read
// them; the error codes follow the PngDec/PngEnc and JpegEnc numbering (0x8069xxxx / 0x8065xxxx).
// Decoding itself is done by stb_image, which handles baseline and progressive streams.

namespace Libs {

LIB_VERSION("JpegDec", 1, "JpegDec", 1, 1);

namespace JpegDec {

constexpr int32_t JPEG_DEC_ERROR_INVALID_ADDR        = -2140864511; // 0x80650001
constexpr int32_t JPEG_DEC_ERROR_INVALID_SIZE        = -2140864510; // 0x80650002
constexpr int32_t JPEG_DEC_ERROR_INVALID_PARAM       = -2140864509; // 0x80650003
constexpr int32_t JPEG_DEC_ERROR_INVALID_HANDLE      = -2140864508; // 0x80650004
constexpr int32_t JPEG_DEC_ERROR_INVALID_WORK_MEMORY = -2140864507; // 0x80650005
constexpr int32_t JPEG_DEC_ERROR_INVALID_DATA        = -2140864496; // 0x80650010
constexpr int32_t JPEG_DEC_ERROR_UNSUPPORT_IMAGE     = -2140864495; // 0x80650011
constexpr int32_t JPEG_DEC_ERROR_DECODE_ERROR        = -2140864494; // 0x80650012

constexpr uint16_t JPEG_DEC_COLOR_SPACE_GRAYSCALE    = 2;
constexpr uint16_t JPEG_DEC_COLOR_SPACE_YCBCR        = 3;
constexpr uint16_t JPEG_DEC_COLOR_SPACE_CMYK         = 4;
constexpr uint16_t JPEG_DEC_IMAGE_FORMAT_BASELINE    = 0;
constexpr uint16_t JPEG_DEC_IMAGE_FORMAT_PROGRESSIVE = 1;
constexpr uint16_t JPEG_DEC_PIXEL_FORMAT_R8G8B8A8    = 0;
constexpr uint16_t JPEG_DEC_PIXEL_FORMAT_B8G8R8A8    = 1;

struct JpegDecCreateParam {
	uint32_t this_size;
	uint32_t attribute;
	uint32_t max_image_width;
};

struct JpegDecParseParam {
	const void* jpeg_mem_addr;
	uint32_t    jpeg_mem_size;
	uint32_t    decode_option; // passed through unchanged to the decode parameters
};

struct JpegDecDecodeParam {
	const void* jpeg_mem_addr;
	void*       image_mem_addr;
	void*       coef_mem_addr;
	uint32_t    jpeg_mem_size;
	uint32_t    image_mem_size;
	uint32_t    coef_mem_size;
	uint32_t    decode_option;
	uint16_t    pixel_format;
	uint16_t    alpha_value;
	uint32_t    image_pitch;
};

// Titles read width/height, a non-zero color space, the coefficient buffer size to allocate for
// decoding and the create parameters (attribute, max width) to use for this image.
struct JpegDecImageInfo {
	uint32_t image_width;
	uint32_t image_height;
	uint16_t color_space;
	uint16_t image_format;
	uint16_t sampling_type;
	uint16_t reserved0;
	uint32_t reserved1;
	uint32_t coef_mem_size;
	uint32_t suggested_attribute;
	uint32_t suggested_max_image_width;
};

static_assert(sizeof(JpegDecCreateParam) == 12);
static_assert(sizeof(JpegDecParseParam) == 16);
static_assert(sizeof(JpegDecDecodeParam) == 48);
static_assert(offsetof(JpegDecDecodeParam, decode_option) == 0x24);
static_assert(offsetof(JpegDecDecodeParam, pixel_format) == 0x28);
static_assert(offsetof(JpegDecDecodeParam, image_pitch) == 0x2c);
static_assert(sizeof(JpegDecImageInfo) == 32);
static_assert(offsetof(JpegDecImageInfo, coef_mem_size) == 0x14);
static_assert(offsetof(JpegDecImageInfo, suggested_attribute) == 0x18);

struct JpegDecContext {
	uint64_t magic;
	uint32_t attribute;
	uint32_t max_image_width;
};

constexpr uint64_t JPEG_DEC_CONTEXT_MAGIC = 0x4b5954594a504744ull; // KYTYJPGD

struct JpegHeaderInfo {
	uint32_t width;
	uint32_t height;
	uint32_t components;
	bool     progressive;
	uint8_t  h_sampling[3];
	uint8_t  v_sampling[3];
};

// Walks the marker segments up to the first start-of-frame. Returns false on malformed or
// truncated data, or when the stream has no frame header before the first scan.
static bool ParseJpegHeader(const void* data, uint32_t size, JpegHeaderInfo* info) {
	if (data == nullptr || info == nullptr || size < 4u) {
		return false;
	}

	const auto* bytes = static_cast<const uint8_t*>(data);
	if (bytes[0] != 0xffu || bytes[1] != 0xd8u) {
		return false;
	}

	uint32_t pos = 2;
	while (pos + 4u <= size) {
		if (bytes[pos] != 0xffu) {
			return false;
		}

		const uint8_t marker = bytes[pos + 1u];
		if (marker == 0xffu) { // fill byte
			pos++;
			continue;
		}
		if (marker == 0x01u || (marker >= 0xd0u && marker <= 0xd8u)) { // segments without length
			pos += 2u;
			continue;
		}
		if (marker == 0xd9u || marker == 0xdau) { // end of image or scan before any frame header
			return false;
		}

		const uint32_t length = (static_cast<uint32_t>(bytes[pos + 2u]) << 8u) | bytes[pos + 3u];
		if (length < 2u || length > size - pos - 2u) {
			return false;
		}

		const bool start_of_frame = (marker >= 0xc0u && marker <= 0xcfu && marker != 0xc4u &&
		                             marker != 0xc8u && marker != 0xccu);
		if (start_of_frame) {
			const uint8_t* frame = bytes + pos + 4u;
			if (length < 8u) {
				return false;
			}

			info->height     = (static_cast<uint32_t>(frame[1]) << 8u) | frame[2];
			info->width      = (static_cast<uint32_t>(frame[3]) << 8u) | frame[4];
			info->components = frame[5];
			info->progressive =
			    (marker == 0xc2u || marker == 0xc6u || marker == 0xcau || marker == 0xceu);

			if (info->width == 0 || info->height == 0 || info->components == 0 ||
			    length < 8u + 3u * info->components) {
				return false;
			}

			for (uint32_t c = 0; c < 3u; c++) {
				const bool present  = c < info->components;
				info->h_sampling[c] = present ? frame[7u + 3u * c] >> 4u : 0;
				info->v_sampling[c] = present ? frame[7u + 3u * c] & 0xfu : 0;
			}

			return true;
		}

		pos += 2u + length;
	}

	return false;
}

static uint16_t ColorSpace(const JpegHeaderInfo& header) {
	switch (header.components) {
		case 1: return JPEG_DEC_COLOR_SPACE_GRAYSCALE;
		case 4: return JPEG_DEC_COLOR_SPACE_CMYK;
		default: return JPEG_DEC_COLOR_SPACE_YCBCR;
	}
}

// Encodes the luma sampling factors relative to chroma: 0 = 4:4:4 (or grayscale),
// 1 = 4:2:2, 2 = 4:2:0, 3 = 4:4:0, 4 = anything else.
static uint16_t SamplingType(const JpegHeaderInfo& header) {
	if (header.components < 3) {
		return 0;
	}
	const uint32_t h = header.h_sampling[0];
	const uint32_t v = header.v_sampling[0];
	if (header.h_sampling[1] != 1 || header.v_sampling[1] != 1 || header.h_sampling[2] != 1 ||
	    header.v_sampling[2] != 1) {
		return 4;
	}
	if (h == 1 && v == 1) {
		return 0;
	}
	if (h == 2 && v == 1) {
		return 1;
	}
	if (h == 2 && v == 2) {
		return 2;
	}
	if (h == 1 && v == 2) {
		return 3;
	}
	return 4;
}

static void FillImageInfo(const JpegHeaderInfo& header, JpegDecImageInfo* image_info) {
	if (image_info != nullptr) {
		*image_info                     = {};
		image_info->image_width         = header.width;
		image_info->image_height        = header.height;
		image_info->color_space         = ColorSpace(header);
		image_info->image_format        = (header.progressive ? JPEG_DEC_IMAGE_FORMAT_PROGRESSIVE
		                                                      : JPEG_DEC_IMAGE_FORMAT_BASELINE);
		image_info->sampling_type       = SamplingType(header);
		image_info->coef_mem_size       = 0; // the host decoder needs no guest buffer
		image_info->suggested_attribute = 0;
		image_info->suggested_max_image_width = header.width;
	}
}

static int32_t KYTY_SYSV_ABI JpegDecQueryMemorySize(const JpegDecCreateParam* param) {
	PRINT_NAME();

	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	LOGF("\t this_size       = %" PRIu32 "\n", param->this_size);
	LOGF("\t attribute       = %" PRIu32 "\n", param->attribute);
	LOGF("\t max_image_width = %" PRIu32 "\n", param->max_image_width);

	if (param->this_size != sizeof(JpegDecCreateParam)) {
		return JPEG_DEC_ERROR_INVALID_SIZE;
	}

	return sizeof(JpegDecContext);
}

static int32_t KYTY_SYSV_ABI JpegDecCreate(const JpegDecCreateParam* param, void* memory_address,
                                           uint32_t memory_size, void** handle) {
	PRINT_NAME();

	if (param == nullptr || memory_address == nullptr || handle == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	LOGF("\t memory_address  = %p\n", memory_address);
	LOGF("\t memory_size     = %" PRIu32 "\n", memory_size);

	if (param->this_size != sizeof(JpegDecCreateParam)) {
		return JPEG_DEC_ERROR_INVALID_SIZE;
	}

	if (memory_size < sizeof(JpegDecContext)) {
		return JPEG_DEC_ERROR_INVALID_WORK_MEMORY;
	}

	auto* ctx            = static_cast<JpegDecContext*>(memory_address);
	ctx->magic           = JPEG_DEC_CONTEXT_MAGIC;
	ctx->attribute       = param->attribute;
	ctx->max_image_width = param->max_image_width;
	*handle              = ctx;

	return 0;
}

static int32_t KYTY_SYSV_ABI JpegDecDecode(void* handle, const JpegDecDecodeParam* param,
                                           JpegDecImageInfo* image_info) {
	PRINT_NAME();

	if (handle == nullptr) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}

	const auto* ctx = static_cast<const JpegDecContext*>(handle);
	if (ctx->magic != JPEG_DEC_CONTEXT_MAGIC) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}

	if (param == nullptr || param->jpeg_mem_addr == nullptr || param->image_mem_addr == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	LOGF("\t jpeg_mem_addr  = %p\n", param->jpeg_mem_addr);
	LOGF("\t image_mem_addr = %p\n", param->image_mem_addr);
	LOGF("\t jpeg_mem_size  = %" PRIu32 "\n", param->jpeg_mem_size);
	LOGF("\t image_mem_size = %" PRIu32 "\n", param->image_mem_size);
	LOGF("\t decode_option  = 0x%08" PRIx32 "\n", param->decode_option);
	LOGF("\t pixel_format   = %" PRIu16 "\n", param->pixel_format);
	LOGF("\t alpha_value    = %" PRIu16 "\n", param->alpha_value);
	LOGF("\t image_pitch    = %" PRIu32 "\n", param->image_pitch);

	if (param->pixel_format != JPEG_DEC_PIXEL_FORMAT_R8G8B8A8 &&
	    param->pixel_format != JPEG_DEC_PIXEL_FORMAT_B8G8R8A8) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}

	if (param->jpeg_mem_size > static_cast<uint32_t>(INT32_MAX)) {
		return JPEG_DEC_ERROR_INVALID_SIZE;
	}

	JpegHeaderInfo header {};
	if (!ParseJpegHeader(param->jpeg_mem_addr, param->jpeg_mem_size, &header)) {
		return JPEG_DEC_ERROR_INVALID_DATA;
	}

	if (header.components != 1 && header.components != 3 && header.components != 4) {
		return JPEG_DEC_ERROR_UNSUPPORT_IMAGE;
	}

	FillImageInfo(header, image_info);

	const uint64_t min_pitch = static_cast<uint64_t>(header.width) * 4u;
	const uint64_t pitch     = (param->image_pitch == 0 ? min_pitch : param->image_pitch);
	const uint64_t min_size  = pitch * (header.height - 1u) + min_pitch;

	if (pitch < min_pitch || min_size > param->image_mem_size) {
		return JPEG_DEC_ERROR_INVALID_SIZE;
	}

	int   width      = 0;
	int   height     = 0;
	int   components = 0;
	auto* rgba = stbi_load_from_memory(static_cast<const stbi_uc*>(param->jpeg_mem_addr),
	                                   static_cast<int>(param->jpeg_mem_size), &width, &height,
	                                   &components, 4);

	if (rgba == nullptr || static_cast<uint32_t>(width) != header.width ||
	    static_cast<uint32_t>(height) != header.height) {
		LOGF("\t decode failed: %s\n", rgba == nullptr ? stbi_failure_reason() : "size mismatch");
		stbi_image_free(rgba);
		return JPEG_DEC_ERROR_DECODE_ERROR;
	}

	const auto alpha    = static_cast<uint8_t>(std::min<uint16_t>(param->alpha_value, 255u));
	const bool bgra     = (param->pixel_format == JPEG_DEC_PIXEL_FORMAT_B8G8R8A8);
	auto*      dst_row  = static_cast<uint8_t*>(param->image_mem_addr);
	const auto row_size = static_cast<size_t>(min_pitch);

	for (uint32_t y = 0; y < header.height; y++) {
		const auto* src = rgba + static_cast<size_t>(y) * row_size;
		auto*       dst = dst_row;

		for (uint32_t x = 0; x < header.width; x++) {
			dst[0] = src[bgra ? 2 : 0];
			dst[1] = src[1];
			dst[2] = src[bgra ? 0 : 2];
			dst[3] = alpha;
			src += 4;
			dst += 4;
		}

		dst_row += pitch;
	}

	stbi_image_free(rgba);

	return (header.width > 32767u || header.height > 32767u
	            ? 0
	            : static_cast<int32_t>((header.width << 16u) | header.height));
}

static int32_t KYTY_SYSV_ABI JpegDecDelete(void* handle) {
	PRINT_NAME();

	if (handle == nullptr) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}

	auto* ctx = static_cast<JpegDecContext*>(handle);
	if (ctx->magic != JPEG_DEC_CONTEXT_MAGIC) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}

	ctx->magic = 0;
	return 0;
}

static int32_t KYTY_SYSV_ABI JpegDecParseHeader(const JpegDecParseParam* param,
                                                JpegDecImageInfo*        image_info) {
	PRINT_NAME();

	if (param == nullptr || image_info == nullptr || param->jpeg_mem_addr == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	JpegHeaderInfo header {};
	if (!ParseJpegHeader(param->jpeg_mem_addr, param->jpeg_mem_size, &header)) {
		return JPEG_DEC_ERROR_INVALID_DATA;
	}

	FillImageInfo(header, image_info);

	LOGF("\t decode_option = 0x%08" PRIx32 "\n", param->decode_option);
	LOGF("\t image_width   = %" PRIu32 "\n", image_info->image_width);
	LOGF("\t image_height  = %" PRIu32 "\n", image_info->image_height);
	LOGF("\t color_space   = %" PRIu16 "\n", image_info->color_space);
	LOGF("\t image_format  = %" PRIu16 "\n", image_info->image_format);
	LOGF("\t sampling_type = %" PRIu16 "\n", image_info->sampling_type);

	return 0;
}

} // namespace JpegDec

LIB_DEFINE(InitJpegDec_1) {
	PRINT_NAME_ENABLE(true);

	LIB_FUNC("uNAUmANZMEw", JpegDec::JpegDecQueryMemorySize);
	LIB_FUNC("JPh3Zgg0Zwc", JpegDec::JpegDecCreate);
	LIB_FUNC("1kzQRoWEgSA", JpegDec::JpegDecDecode);
	LIB_FUNC("Hwh11+m5KoI", JpegDec::JpegDecDelete);
	LIB_FUNC("LSinoSQH790", JpegDec::JpegDecParseHeader);
}

} // namespace Libs
