#include <float.h>
#include "cuda_func.h"
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile.cuh"
#define GGML_ASSERT(x) assert(x)
#define GGML_ABORT(...)

void ggml_cuda_flash_attn_ext_tile(const flash_attn_ext_context& ctx)
{
    switch (ctx.K.ne0) {
    case  40: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case< 40, 40>(ctx);
    } break;
    case  72: {
        GGML_ASSERT(ctx.V.ne0 == ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case< 72, 72>(ctx);
    } break;
    case  64: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case< 64, 64>(ctx);
    } break;
    case  80: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case< 80, 80>(ctx);
    } break;
    case  96: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case< 96, 96>(ctx);
    } break;
    case 112: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx);
    } break;
    case 128: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx);
    } break;
    case 256: {
        GGML_ASSERT(ctx.V.ne0 ==ctx.K.ne0);
        ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx);
    } break;
    case 576: {
        GGML_ASSERT(ctx.V.ne0 == 512);
        ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx);
    } break;
    default: {
        GGML_ABORT("Unsupported head size");
    } break;
    }
}