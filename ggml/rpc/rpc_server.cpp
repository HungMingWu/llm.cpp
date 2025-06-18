module;
#include <assert.h>
#include <stdint.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <vector>
#include <unordered_map>

#define GGML_ASSERT(...) assert(__VA_ARGS__)

namespace fs = std::filesystem;

#define GGML_PRINT_DEBUG(...)
#define GGML_LOG_ERROR(...)

module ggml;
import :rpc.ds;
import :rpc.helper;
import :rpc.server;

void rpc_server::hello(rpc_msg_hello_rsp& response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    GGML_PRINT_DEBUG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req& request, rpc_msg_get_alloc_size_rsp& response) {
    ggml_backend_buffer_type_t buft;

    ggml_context ctx;
    ggml_tensor* tensor = deserialize_tensor(&ctx, &request.tensor);

    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }

    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = backend->get_default_buffer_type();
    }
    else {
        buft = tensor->buffer->get_type();
    }

    response.alloc_size = buft->get_alloc_size(tensor);
    return true;
}

void rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req& request, rpc_msg_alloc_buffer_rsp& response) {
    ggml_backend_buffer_type_t buft = backend->get_default_buffer_type();
    std::unique_ptr<ggml_backend_buffer> buffer = buft->alloc_buffer(request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (!buffer) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer.get());
        response.remote_size = buffer->get_size();
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n", __func__, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer.release());
    }
    else {
        GGML_LOG_ERROR("[%s] size: %" PRIu64 " -> failed\n", __func__, request.size);
    }
}

void rpc_server::get_alignment(rpc_msg_get_alignment_rsp& response) {
    ggml_backend_buffer_type_t buft = backend->get_default_buffer_type();
    size_t alignment = buft->get_alignment();
    GGML_PRINT_DEBUG("[%s] alignment: %lu\n", __func__, alignment);
    response.alignment = alignment;
}

void rpc_server::get_max_size(rpc_msg_get_max_size_rsp& response) {
    ggml_backend_buffer_type_t buft = backend->get_default_buffer_type();
    size_t max_size = buft->get_max_size();
    GGML_PRINT_DEBUG("[%s] max_size: %lu\n", __func__, max_size);
    response.max_size = max_size;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req& request, rpc_msg_buffer_get_base_rsp& response) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void* base = buffer->get_base();
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req& request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    delete buffer;
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req& request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    buffer->clear(request.value);
    return true;
}

ggml_tensor* rpc_server::deserialize_tensor(struct ggml_context* ctx, const rpc_tensor* tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor* result = ctx->create((ggml_type)tensor->type,
        { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] });

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t)result->nbytes();
        uint64_t buffer_start = (uint64_t)result->buffer->get_base();
        uint64_t buffer_size = (uint64_t)result->buffer->get_size();
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op)tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void*>(tensor->data);
    result->set_name(tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t>& input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor* in_tensor = (const rpc_tensor*)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    ggml_context ctx;
    ggml_tensor* tensor = deserialize_tensor(&ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t)tensor->buffer->get_base();
        const size_t p1 = p0 + tensor->buffer->get_size();

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size >(p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void* data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        std::string hash_str = std::format("{:016}", hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char*)data, size);
        printf("[%s] saved to '%s'\n", __func__, cache_file.c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t>& data) {
    if (!cache_dir) {
        return false;
    }
    std::string hash_str = std::format("{:016}", hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    if (!fs::exists(cache_file)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char*)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req& request, rpc_msg_set_tensor_hash_rsp& response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    ggml_context ctx;
    ggml_tensor* tensor = deserialize_tensor(&ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
        __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t)tensor->buffer->get_base();
        const size_t p1 = p0 + tensor->buffer->get_size();

        if (request.tensor.data + request.offset < p0
            || request.tensor.data + request.offset >= p1
            || size >(p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req& request) {
    ggml_context ctx;
    ggml_tensor* tensor = deserialize_tensor(&ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }

    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer) {
        buffer->init_tensor(tensor);
    }
    else {
        GGML_LOG_ERROR("Null buffer for tensor passed to init_tensor function\n");
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req& request, std::vector<uint8_t>& response) {
    ggml_context ctx;
    ggml_tensor* tensor = deserialize_tensor(&ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t)tensor->buffer->get_base();
        const size_t p1 = p0 + tensor->buffer->get_size();

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size >(p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                __func__, request.tensor.data, request.offset, request.size, p0, p1);
            return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req& request, rpc_msg_copy_tensor_rsp& response) {
    ggml_context ctx;

    ggml_tensor* src = deserialize_tensor(&ctx, &request.src);
    ggml_tensor* dst = deserialize_tensor(&ctx, &request.dst);
    if (src == nullptr || dst == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size = (uint64_t)src->nbytes();
    uint64_t dst_data = (uint64_t)dst->data;
    uint64_t dst_base = (uint64_t)dst->buffer->get_base();
    uint64_t dst_buf_sz = (uint64_t)dst->buffer->get_size();

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_PRINT_DEBUG("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
            "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
            "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
            __func__,
            dst_data,
            dst_data + src_size,
            dst_base,
            dst_base + dst_buf_sz);
        return false;
    }

    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n",
        __func__, (void*)src->buffer, (void*)dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor* rpc_server::create_node(uint64_t id,
    struct ggml_context* ctx,
    const std::unordered_map<uint64_t, const rpc_tensor*>& tensor_ptrs,
    std::unordered_map<uint64_t, struct ggml_tensor*>& tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor* tensor = it_ptr->second;

    struct ggml_tensor* result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        }
        else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                    __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    }
    else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t>& input, rpc_msg_graph_compute_rsp& response) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    if (input.size() < sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t* nodes = (const uint64_t*)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor* tensors = (const rpc_tensor*)(input.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(n_tensors));
    GGML_PRINT_DEBUG("[%s] n_nodes: %u, n_tensors: %u\n", __func__, n_nodes, n_tensors);

    ggml_context ctx;
    ggml_cgraph graph;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph.nodes.push_back(create_node(id, &ctx, tensor_ptrs, tensor_map));

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph.nodes.back() == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    ggml_status status = backend->graph_compute(&graph);
    response.result = status;
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        delete buffer;
    }
}
