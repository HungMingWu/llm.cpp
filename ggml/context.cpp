module;
#include <cstdlib>
#include <numeric>
#include <span>
#include <string_view>
#include "inplace_vector.hpp"
#define GGML_ASSERT(...)

module ggml;
import :alloc;
import :ds;

ggml_context::ggml_context()
{
}

ggml_context::~ggml_context()
{
}

ggml_tensor* ggml_context::create_new_tensor_impl(
    ggml_type type,
    std::span<int64_t> ne,
    ggml_tensor* view_src,
    size_t  view_offs)
{
    GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != nullptr && view_src->view_src != nullptr) {
        view_offs += view_src->view_offs;
        view_src = view_src->view_src;
    }

    GGML_ASSERT(ne[0] % ggml_blck_size(type) == 0);
    size_t data_size =
        std::accumulate(ne.begin(), ne.end(), ggml_type_size(type), std::multiplies<size_t>())
        / ggml_blck_size(type);

    GGML_ASSERT(view_src == nullptr || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    ggml_tensor* new_tensor = new ggml_tensor();
    new_tensor->type = type;

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(new_tensor->data);

    for (int i = 0; i < ne.size(); i++) {
        new_tensor->ne[i] = ne[i];
    }

    new_tensor->nb[0] = ggml_type_size(type);
    new_tensor->nb[1] = new_tensor->nb[0] * (new_tensor->ne[0] / ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = new_tensor->nb[i - 1] * new_tensor->ne[i - 1];
    }
    tensors.push_back(new_tensor);
    return new_tensor;
}

ggml_tensor* ggml_context::create(ggml_type type, std::initializer_list<int64_t> ne)
{
	GGML_ASSERT(ne.size() < 5);
    cpp26::inplace_vector<int64_t, 4> vec(ne.begin(), ne.end());
	return create_new_tensor_impl(type, vec, nullptr, 0);
}

ggml_tensor* ggml_context::create(ggml_type type, std::initializer_list<int64_t> ne, ggml_tensor* view_src, size_t view_offset)
{
    ggml_tensor* result = create(type, ne);
    result->view_src = view_src;
    result->view_offs = view_offset;
    return result;
}

ggml_tensor* ggml_context::find(std::string_view name)
{
	for (auto& tensor : tensors) {
		if (tensor->name == name) {
			return tensor;
		}
	}
    return nullptr;
}
