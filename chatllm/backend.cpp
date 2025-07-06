module;
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdarg.h>

#include "basics.h"

#ifndef GGML_USE_CPU
#ifndef GGML_BACKEND_DL
//#define GGML_BACKEND_DL
#endif
#endif

extern void log_internal(int level, const char* text);

module chatllm;
import ggml;

namespace chatllm
{
    static std::string str_format(const char* format, va_list args)
    {
        // First, determine the required length of the formatted string
        va_list args_copy;
        va_copy(args_copy, args);
        int length = std::vsnprintf(nullptr, 0, format, args_copy);
        va_end(args_copy);

        if (length < 0) {
            return "";
        }

        std::vector<char> buffer(length + 1);

        std::vsnprintf(buffer.data(), buffer.size(), format, args);

        return std::string(buffer.data());
    }

    void ggml::log(enum ggml_log_level level, const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        auto s = str_format(format, args);
        log_internal(level, s.c_str());
        va_end(args);
    }

    void* BackendBuffer::get_base(void)
    {
        return buf->get_base();
    }

    size_t BackendBuffer::get_size(void) const
    {
        return buf->get_size();
    }

    bool BackendBuffer::is_host(void)
    {
        return buf->is_host();
    }

    BackendBuffer::~BackendBuffer()
    {
        delete buf;
    }

    void BackendBuffer::assign_to(ggml::tensor* tensor, size_t offset)
    {
        uint8_t* data = (uint8_t*)get_base() + offset;
        ggml_backend_tensor_alloc(buf, tensor, data);
    }

    BackendBuffer::BackendBuffer(ggml_backend_buffer_t buf)
        : buf(buf)
    {
    }

    void BackendBufAllocator::show_info(void)
    {
        ggml::log(GGML_LOG_LEVEL_INFO, "%30s allocated buffer size = (%8.2f, %8.2f) MiB\n", backend->backend->get_name(), total[0] / 1024.0 / 1024.0, total[1] / 1024.0 / 1024.0);
    }

    Backend* BackendBufAllocator::get_backend(void)
    {
        return backend;
    }

    LayerBufAllocator::LayerBufAllocator() : LayerBufAllocator(nullptr, nullptr, nullptr) {}
    LayerBufAllocator::LayerBufAllocator(ggml_backend_allocator alloc, Backend* backend) : LayerBufAllocator(alloc, alloc, backend) {}
    LayerBufAllocator::LayerBufAllocator(ggml_backend_allocator alloc_matrix, ggml_backend_allocator alloc_others, Backend* backend)
        : BackendBufAllocator(backend), alloc_matrix(alloc_matrix), alloc_others(alloc_others)
    {
        CHATLLM_CHECK(alloc_matrix == alloc_others) << " TODO: alloc_matrix must be alloc_others now.";
    }

    void LayerBufAllocator::show_info(void)
    {
        BackendBufAllocator::show_info();
        ggml::log(GGML_LOG_LEVEL_INFO, "\tMatrix = %s, Others = %s\n", get_allocator(Usage::Matrix)->get_name(), get_allocator(Usage::Others)->get_name());
    }

    BackendBuffer* LayerBufAllocator::alloc(size_t size, Usage usage)
    {
        total[usage] += size;
        std::unique_ptr<ggml_backend_buffer> buf = get_allocator(usage)->alloc_buffer(size);

        CHATLLM_CHECK(buf) << __FUNCTION__ << "() failed to allocate buffer of size " << size;

        auto r = new BackendBuffer(buf.release());
        buffers.emplace_back(r);
        return r;
    }

    bool LayerBufAllocator::alloc(ggml::tensor* tensor, Usage usage)
    {
        BackendBuffer* buf = alloc(get_alloc_size(tensor), usage);
        if (nullptr == buf) return false;

        buf->assign_to(tensor);
        return true;
    }

    bool LayerBufAllocator::alloc(ggml::tensor* tensor)
    {
        return alloc(tensor, detect_usage(tensor));
    }

    size_t  LayerBufAllocator::get_alloc_size(ggml::tensor* tensor, Usage usage)
    {
        return get_allocator(usage)->get_alloc_size(tensor);
    }

    size_t  LayerBufAllocator::get_alloc_size(ggml::tensor* tensor)
    {
        return get_alloc_size(tensor, detect_usage(tensor));
    }

    bool LayerBufAllocator::supported_by_backend(Backend* backend, ggml::tensor* tensor)
    {
        ggml_backend_allocator allocator = get_allocator(tensor); return false;
        return backend->backend->supports_buft(allocator);
    }

    BackendBufAllocator::Usage LayerBufAllocator::detect_usage(ggml::tensor* tensor)
    {
        int dims = ggml::n_dims(tensor);
        return dims >= 2 ? Usage::Matrix : Usage::Others;
    }

    ggml_backend_allocator LayerBufAllocator::get_allocator(Usage usage)
    {
        switch (usage)
        {
        case Usage::Matrix:
            return alloc_matrix;
        case Usage::Others:
            return alloc_others;
        default:
            CHATLLM_CHECK(false);
            return nullptr;
        }
    }

    ggml_backend_allocator LayerBufAllocator::get_allocator(ggml::tensor* tensor)
    {
        return get_allocator(detect_usage(tensor));
    }

    size_t LayerBufAllocator::get_alignment(Usage usage) const
    {
        switch (usage)
        {
        case Usage::Matrix:
            return alloc_matrix->get_alignment();
        case Usage::Others:
            return alloc_others->get_alignment();
        default:
            CHATLLM_CHECK(0);
            return 0;
        }
    }

    size_t LayerBufAllocator::get_max_size(Usage usage) const
    {
        switch (usage)
        {
        case Usage::Matrix:
            return alloc_matrix->get_max_size();
        case Usage::Others:
            return alloc_others->get_max_size();
        default:
            CHATLLM_CHECK(0);
            return 0;
        }
    }

    void LayerBufAllocator::free_all_buffers(void)
    {
        memset(&total, 0, sizeof(total));
        buffers.clear();
    }

    bool LayerBufAllocator::operator ==(const LayerBufAllocator& b)
    {
        return (alloc_matrix == b.alloc_matrix)
            && (alloc_others == b.alloc_others)
            && (backend == b.backend);
    }


    void LayerAllocatorManager::set_misc_layer_backend_mapping(int prolog, int epilog)
    {
        prolog_layer_backend_map_to_layer_id = prolog;
        epilog_layer_backend_map_to_layer_id = epilog;
    }

    void LayerAllocatorManager::move_to_layer(int layer_id)
    {
        cur_layer = layer_id;
    }

    int LayerAllocatorManager::get_cur_layer(void) const
    {
        return cur_layer;
    }

    LayerBufAllocator* LayerAllocatorManager::get_allocator(void)
    {
        auto id = get_mapped_layer_id(cur_layer);
        return &allocators[id];
    }

    LayerBufAllocator* LayerAllocatorManager::get_allocator(int layer_id)
    {
        auto id = get_mapped_layer_id(layer_id);
        return &allocators[id];
    }

    LayerBufAllocator* LayerAllocatorManager::get_allocator(ggml::tensor* tensor)
    {
        return alloc_of_tensor[tensor];
    }

    void LayerAllocatorManager::register_tensor_allocator(ggml::tensor* tensor, LayerBufAllocator* allocator)
    {
        alloc_of_tensor.insert_or_assign(tensor, allocator);
    }

    void LayerAllocatorManager::override_to_cpu_only(bool flag)
    {
        cpu_override = flag;
    }

    int LayerAllocatorManager::get_mapped_layer_id(int layer_id)
    {
        int id = layer_id;
        switch (id)
        {
        case MiscLayer::Prolog:
            id = prolog_layer_backend_map_to_layer_id;
            break;
        case MiscLayer::Epilog:
            id = epilog_layer_backend_map_to_layer_id;
            //if (id < 0) id = (int)allocators.size() - 2;
            break;
        default:
            break;
        }
        if (cpu_override || (id < 0) || (id >= (int)allocators.size()))
            id = (int)allocators.size() - 1;

        return id;
    }

    ggml_backend_reg_t ComputeManager::backend_rpc = nullptr;

    void ComputeManager::init(const std::string& ggml_dir)
    {
#ifdef GGML_BACKEND_DL
        static bool initialized = false;
        if (initialized) return;
        initialized = true;
        ggml::log(GGML_LOG_LEVEL_INFO, "loading backends...");
        ggml_backend_load_all_from_path(ggml_dir.size() > 0 ? ggml_dir.c_str() : nullptr);
#endif

        if (ggml_backend_reg_count() < 1)
        {
            ggml::log(GGML_LOG_LEVEL_ERROR, "FATAL: no backend is loaded. check dll directories?");
            exit(-1);
        }

        for (int i = 0; i < (int)ggml_backend_reg_count(); i++)
        {
            auto reg = ggml_backend_reg_get(i);
            if (reg->get_name() == "RPC")
            {
                ComputeManager::backend_rpc = reg;
                break;
            }
        }
    }

    std::string ComputeManager::dev_type_to_str(DeviceType type)
    {
        switch (type)
        {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::GPU:
            return "GPU";
        case DeviceType::ACCEL:
            return "ACCEL";
        default:
            return "UNKNOWN";
        }
    }

    ggml_backend_allocator ComputeManager::get_default_allocator_cpu(bool host_buffer, int gpu_id)
    {
        ggml_backend_allocator allocator = nullptr;

        if (0) // gpu_id >= 0), TO FIX
        {
            auto dev = ggml_backend_dev_get(gpu_id);
            if (dev)
                allocator = dev->get_host_buffer_type();
        }

        if (allocator == nullptr)
            allocator = ggml_backend_cpu_buffer_type();

        return allocator;
    }

    ggml_backend_allocator ComputeManager::get_default_allocator(ggml_backend_t backend)
    {
        return backend->get_default_buffer_type();
    }

    int ComputeManager::get_device_count(void)
    {
        ComputeManager::init();
        return (int)ggml_backend_dev_count();
    }

    ggml_backend_t ComputeManager::init_backend_device(int index, const char* param)
    {
        auto dev = ggml_backend_dev_get(index);
        return dev ? dev->init_backend(param) : nullptr;
    }

    ggml_backend_allocator ComputeManager::get_default_allocator_offload(int device)
    {
        ggml_backend_allocator allocator = nullptr;

        auto dev = ggml_backend_dev_get(device);
        if (dev)
            allocator = dev->get_buffer_type();

        if (allocator == nullptr)
            allocator = get_default_allocator_cpu(true, true);
        return allocator;
    }

    size_t ComputeManager::get_device_free_memory(int device, size_t* p_total)
    {
        size_t total = 0;
        size_t free = 0;

        auto dev = ggml_backend_dev_get(device);
        if (dev)
            dev->get_memory(&free, &total);

        if (p_total) *p_total = total;
        return free;
    }

    bool ComputeManager::get_device_info(int device, DeviceInfo& info)
    {
        auto dev = ggml_backend_dev_get(device);

        if (nullptr == dev)
            return false;

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        info.type = (DeviceType)props.type;
        info.backend_name = dev->get_backend_reg()->get_name();
        info.name = dev->get_name();
        info.description = utils::trim(dev->get_description());
        info.total_memory = props.memory_total;
        info.free_memory = props.memory_free;
        return true;
    }

    void ComputeManager::get_devices_info(std::vector<DeviceInfo>& devs)
    {
        ComputeManager::init();
        devs.clear();
        for (int i = 0; i < get_device_count(); i++)
        {
            DeviceInfo info;
            CHATLLM_CHECK(get_device_info(i, info)) << __func__ << ": failed to get device #" << i;
            devs.push_back(info);
        }
    }

    bool ComputeManager::prepare_rpc_devices(const std::string& endpoints)
    {
#if 0
        if (endpoints.size() < 1) return true;

        if (!ComputeManager::backend_rpc)
        {
            ggml::log(GGML_LOG_LEVEL_ERROR, "%s: RPC backend not available", __FUNCTION__);
            return false;
        }

        auto rpc_add_device =
            (ggml_backend_rpc_add_device_t)ggml_backend_reg_get_proc_address(ComputeManager::backend_rpc,
                "ggml_backend_rpc_add_device");
        CHATLLM_CHECK(rpc_add_device) << __FUNCTION__ << ": ggml_backend_rpc_add_device() not found in RPC backend";

        std::string s(endpoints);
        while (s.size() > 0)
        {
            auto pos = s.find(';');
            std::string endpoint = s.substr(0, pos);

            if (endpoint.find(':') == std::string::npos)
                endpoint = "127.0.0.1:" + endpoint;

            rpc_add_device(endpoint.c_str());

            if (pos == std::string::npos) break;
            s = s.substr(pos + 1);
        }
#endif
        return true;
    }

    bool ComputeManager::start_rpc_server(int device, const char* endpoint, size_t backend_mem, const char* cache_dir)
    {
#if 0
        if (!ComputeManager::backend_rpc)
        {
            ggml::log(GGML_LOG_LEVEL_ERROR, "%s: RPC backend not available", __FUNCTION__);
            return false;
        }

        if (cache_dir == nullptr)
        {

        }

        auto rpc_start_server =
            (ggml_backend_rpc_start_server_t)ggml_backend_reg_get_proc_address(ComputeManager::backend_rpc,
                "ggml_backend_rpc_start_server");
        CHATLLM_CHECK(rpc_start_server) << __FUNCTION__ << ": ggml_backend_rpc_start_server() not found in RPC backend";

        DeviceInfo dev;
        if (!get_device_info(device, dev))
            return false;

        auto backend = ComputeManager::init_backend_device(device);
        if (nullptr == backend)
            return false;

        if (backend_mem <= 0)
            backend_mem = dev.free_memory;

        std::string s(endpoint);
        if (s.find(':') == std::string::npos)
            s = "0.0.0.0:" + s;

        ggml::log(GGML_LOG_LEVEL_INFO, "    type        : %s\n", ComputeManager::dev_type_to_str(dev.type).c_str());
        ggml::log(GGML_LOG_LEVEL_INFO, "    cache dir   : %s\n", cache_dir)
        ggml::log(GGML_LOG_LEVEL_INFO, "%s - %s (%s)\n", dev.backend_name.c_str(), dev.name.c_str(), dev.description.c_str());
        ggml::log(GGML_LOG_LEVEL_INFO, "    type: %s\n", ComputeManager::dev_type_to_str(dev.type).c_str());
        ggml::log(GGML_LOG_LEVEL_INFO, "    memory total: %zd B\n", dev.total_memory);
        ggml::log(GGML_LOG_LEVEL_INFO, "    memory free : %zd B\n", dev.free_memory);

        rpc_start_server(backend, s.c_str(), cache_dir, backend_mem, backend_mem);
        delete backend;
#endif
        return true;
    }

    Backend::Backend(ggml_backend_t backend, int n_layers, bool use_gpu)
        : backend(backend), n_layers(n_layers), use_gpu(use_gpu)
    {
        // FIXME: find a better way
        _is_cpu = backend->get_name() == std::string_view("CPU");
    }

    bool Backend::is_cpu(void) const
    {
        return _is_cpu;
    }

    ggml_backend_allocator Backend::get_allocator(BufferType bt)
    {
        if (is_cpu() || (BufferType::Shared == bt) || !use_gpu)
        {
            // use host buffers for the CPU backend compute buffer
            return ComputeManager::get_default_allocator_cpu(true, use_gpu);
        }
        else
        {
            return ComputeManager::get_default_allocator(backend.get());
        }
    }

    void Backend::write_tensor_data_async(ggml::tensor* tensor, const void* data, size_t offset, size_t size)
    {
        backend->set_tensor_async(tensor, data, offset, size);
    }

    void Backend::write_tensor_data(ggml::tensor* tensor, const void* data, size_t offset, size_t size)
    {
        ggml_backend_tensor_set(tensor, data, offset, size);
    }

    void Backend::write_tensor_data(ggml::tensor* tensor, const void* data)
    {
        ggml_backend_tensor_set(tensor, data, 0, tensor->nbytes());
    }

    void Backend::read_tensor_data_async(ggml::tensor* tensor, void* data, size_t offset, size_t size)
    {
        backend->get_tensor_async(tensor, data, offset, size);
    }

    void Backend::read_tensor_data(ggml::tensor* tensor, void* data, size_t offset, size_t size)
    {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }

    void Backend::read_tensor_data(ggml::tensor* tensor, void* data)
    {
        ggml_backend_tensor_get(tensor, data, 0, tensor->nbytes());
    }

    void Backend::synchronize(void)
    {
        backend->synchronize();
    }

    BackendContext::BackendContext()
    {
        ComputeManager::init();
    }

    bool BackendContext::is_using_gpu(void) const
    {
        return backends.size() > 1;
    }

    static bool parse_gpu_cfg(BackendContext::gpu_cfg& cfg, const std::string& s)
    {
        cfg.id = 0;
        cfg.n_layers = -1;
        cfg.epilog = false;
        cfg.prolog = false;

        std::string t(s);

        size_t pos = t.find_first_of(':');
        std::string part = t.substr(0, pos);
        if (pos != std::string::npos)
        {
            cfg.id = atoi(part.c_str());
            t = t.substr(pos + 1);
        }

        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(',');
            part = t.substr(0, pos);

            if (part.size() > 0)
            {
                if (part.compare("all") == 0)
                {
                    cfg.prolog = true;
                    cfg.epilog = true;
                    cfg.n_layers = 99999;
                }
                else if (part.compare("prolog") == 0)
                    cfg.prolog = true;
                else if (part.compare("epilog") == 0)
                    cfg.epilog = true;
                else
                    cfg.n_layers = std::max(atoi(part.c_str()), 0);
            }

            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }

        return (cfg.n_layers >= 1) || cfg.prolog || cfg.epilog;
    }

    static int index_of_gpu_cfg(const std::vector<BackendContext::gpu_cfg>& gpu_cfgs, int id)
    {
        for (int i = 0; i < (int)gpu_cfgs.size(); i++)
            if (gpu_cfgs[i].id == id)
                return i;
        return -1;
    }

    static bool parse_gpu_layers(std::vector<BackendContext::gpu_cfg>& gpu_cfgs, const std::string& s)
    {
        std::string t(s);
        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(';');

            BackendContext::gpu_cfg cfg;
            if (parse_gpu_cfg(cfg, t.substr(0, pos)))
            {
                int index = index_of_gpu_cfg(gpu_cfgs, cfg.id);
                if (index >= 0)
                    gpu_cfgs[index].merge(cfg);
                else
                    gpu_cfgs.push_back(cfg);
            }

            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }
        return true;
    }

    std::string BackendContext::get_ngl_of_model(const std::map<std::string, std::string>& model_n_gpu_layers, const std::string& model_id, const std::string fallback_id)
    {
        std::string gpu_cfgs;
        auto k = model_n_gpu_layers.find(model_id);
        if (k == model_n_gpu_layers.end())
            k = model_n_gpu_layers.find(fallback_id);
        if (k != model_n_gpu_layers.end())
            gpu_cfgs = k->second;
        return gpu_cfgs;
    }

    void BackendContext::init(const std::map<std::string, std::string>& model_n_gpu_layers, const std::string& model_id, const int n_layers, const size_t graph_max_nodes_num, const int n_threads, const std::string fallback_id)
    {
        init(get_ngl_of_model(model_n_gpu_layers, model_id, fallback_id), n_layers, graph_max_nodes_num, n_threads);
    }

    void BackendContext::init(const std::string& gpu_cfgs, const int n_layers, const size_t graph_max_nodes_num, const int n_threads)
    {
        std::vector<BackendContext::gpu_cfg> cfgs;
        parse_gpu_layers(cfgs, gpu_cfgs);
        init(cfgs, n_layers, graph_max_nodes_num, n_threads);
    }

    void BackendContext::init(const std::vector<gpu_cfg>& gpu_cfgs, const int n_layers, const size_t graph_max_nodes_num, const int n_threads)
    {
        int prolog_id = -1;
        int epilog_id = -1;
        int n_gpu_layers = 0;
        for (int i = 0; i < (int)gpu_cfgs.size(); i++)
        {
            auto& cfg = gpu_cfgs[i];
            if (cfg.n_layers > 0)
                n_gpu_layers += cfg.n_layers;
            if ((prolog_id < 0) && cfg.prolog)
                prolog_id = i;
            if ((epilog_id < 0) && cfg.epilog)
                epilog_id = i;
        }

        const bool use_gpu = n_gpu_layers > 0;

        buf_compute_meta.resize(ggml_tensor_overhead() * graph_max_nodes_num + ggml_graph_overhead_custom(graph_max_nodes_num, false));

        auto init_device = [this, use_gpu, n_threads](int device, ggml_backend_dev_t dev, int n_layers)
            {
                auto reg = dev->get_backend_reg();

                ggml_backend_t backend = ComputeManager::init_backend_device(device);
                CHATLLM_CHECK(backend != nullptr) << __func__ << ": failed to initialize backend: #" << device;
                backends.emplace_back(backend, n_layers, use_gpu);

#if 0
                if (n_threads > 0)
                {
                    auto set_n_threads = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                    if (set_n_threads)
                        set_n_threads(backend, n_threads);
                }
#endif
            };

        if (use_gpu)
        {
            for (auto cfg : gpu_cfgs)
            {
                int device = cfg.id >= 0 ? cfg.id : 0;
                CHATLLM_CHECK(device < ComputeManager::get_device_count()) << __func__ << ": backend device: #" << device << " out of range";

                auto dev = ggml_backend_dev_get(device);
                CHATLLM_CHECK(dev != nullptr) << __func__ << ": failed to found backend device: #" << device;

                init_device(device, dev, cfg.n_layers);
            }
        }

        // append CPU backend
        {
            int device = ComputeManager::get_device_count() - 1;
            auto dev = ggml_backend_dev_get(device);
            CHATLLM_CHECK(dev != nullptr) << __func__ << ": failed to found CPU device: #" << device;
            CHATLLM_CHECK(dev->get_type() == GGML_BACKEND_DEVICE_TYPE_CPU) << __func__ << ": device #" << device << " is not CPU, but " << dev->get_type();
            init_device(device, dev, n_layers - n_gpu_layers);

            backend_cpu = backends[backends.size() - 1].backend.get();
        }

        host_allocator.alloc_matrix = host_allocator.alloc_others = backends[backends.size() - 1].get_allocator(BufferType::Shared);

        int layer_id = 0;
        for (auto& backend : backends)
        {
            for (int i = 0; (i < backend.n_layers) && (layer_id < n_layers); i++, layer_id++)
            {
                // TODO: matrix and others
                layer_allocators.allocators.emplace_back(backend.get_allocator(BufferType::Dedicated), &backend);
            }
        }

        if (prolog_id >= 0)
        {
            auto& backend = backends[prolog_id];
            prolog_id = (int)layer_allocators.allocators.size();
            layer_allocators.allocators.emplace_back(backend.get_allocator(BufferType::Dedicated), &backend);
        }
        if (epilog_id >= 0)
        {
            auto& backend = backends[epilog_id];
            epilog_id = (int)layer_allocators.allocators.size();
            layer_allocators.allocators.emplace_back(backend.get_allocator(BufferType::Dedicated), &backend);
        }

        layer_allocators.set_misc_layer_backend_mapping(prolog_id, epilog_id);

        // a "faked" layer for CPU
        layer_allocators.allocators.emplace_back(host_allocator.alloc_matrix, &backends[backends.size() - 1]);

        gg_bufts.clear();
        gg_backends.clear();

        for (auto& b : backends)
        {
            gg_backends.push_back(b.backend.get());
            gg_bufts.push_back(b.get_allocator(BufferType::Dedicated));
        }
        sched = std::make_unique<ggml_backend_sched>(gg_backends.data(), gg_bufts.data(), (int)gg_backends.size(), false, false);
    }

    BackendContext::~BackendContext()
    {
        delete buf_output;
    }

    bool BackendContext::reserve_memory(ggml_cgraph* gf)
    {
        return sched->reserve(gf);
    }

    bool BackendContext::alloc_graph(ggml_cgraph* gf)
    {
        return sched->alloc_graph(*gf);
    }

    void BackendContext::compute_graph(ggml_cgraph* gf)
    {
        if (backend_cpu != nullptr)
        {
            auto reg = backend_cpu->get_device()->get_backend_reg();
#if 0
            ggml_backend_set_abort_callback_t set_abort_callback =
                (ggml_backend_set_abort_callback_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            set_abort_callback(backend_cpu, abort_callback, abort_callback_data);
#endif
        }

        if (observe_tensor_callback)
            sched->set_eval_callback([this](ggml::tensor* t, bool ask) {
                if (ask)
                    return need_observe_tensor_callback(t, observe_tensor_callback_data);
                else
                    return observe_tensor_callback(t, observe_tensor_callback_data);
            });
        else
            sched->set_eval_callback(nullptr);

        sched->graph_compute(*gf);
    }

    void BackendContext::reset()
    {
        sched->reset();
    }

    void BackendContext::dump_graph(ggml_cgraph* gf, const char* file_name)
    {
        sched->dump_dot(gf, file_name);
        ggml::log(GGML_LOG_LEVEL_INFO, "dot -Tsvg %s -o %s.svg && open %s.svg\n", file_name, file_name, file_name);
    }

    void BackendContext::set_abort_callback(struct llama_context* ctx, bool (*abort_callback)(void* data), void* abort_callback_data)
    {
        this->abort_callback = abort_callback;
        this->abort_callback_data = abort_callback_data;
    }

    void BackendContext::set_eval_observe_callback(ggml::need_observe_tensor_evaluation_callback need_observe_tensor_callback,
        ggml::observe_tensor_evaluation_callback observe_tensor_callback, void* user_data)
    {
        this->need_observe_tensor_callback = need_observe_tensor_callback;
        this->observe_tensor_callback = observe_tensor_callback;
        this->observe_tensor_callback_data = user_data;
    }

    void BackendContext::show_buffer_sizes(void)
    {
        for (size_t i = 0; i < layer_allocators.allocators.size(); i++)
        {
            ggml::log(GGML_LOG_LEVEL_INFO, "layer #%d", (int)i);
            layer_allocators.allocators[i].show_info();
        }

        for (size_t i = 0; i < gg_backends.size(); i++)
        {
            ggml_backend_buffer_type_t buft = gg_bufts[i];
            size_t size = sched->get_buffer_size(gg_backends[i]);
            ggml::log(GGML_LOG_LEVEL_INFO, "%s: %30s compute buffer size = %8.2f MiB\n", __func__, buft->get_name(), size / 1024.0 / 1024.0);
        }
    }

    void BackendContext::synchronize(void)
    {
        for (auto& backend : backends)
            backend.synchronize();
    }

    ComputeContext::ComputeContext(BackendContext* backend_context) : backend_context(backend_context)
    {
    }

    ggml_cgraph* ComputeContext::get_cgraph(void)
    {
        return nullptr;
    }

    void ComputeContext::cb_new_tensor(ggml::tensor* tensor)
    {
        if (get_backend())
            get_sched()->set_tensor_backend(tensor, get_backend()->backend.get());
        register_tensor_allocator(tensor, backend_context->layer_allocators.get_allocator());
    }

    void ComputeContext::cb_op_tensor(ggml::tensor* tensor)
    {
        if (get_backend() == nullptr) return;
        if (get_backend()->backend->supports_op(tensor))
        {
            //struct ggml_backend_buffer *buffer = tensor->buffer;
            //if (buffer && ggml_backend_supports_buft(get_backend()->backend, ggml_backend_buffer_get_type(buffer)))

            //if (!ggml::is_view_op(tensor))
            //{
            //    ggml_backend_sched_set_tensor_backend(get_sched(), tensor, get_backend()->backend);
            //}
        }
        else
        {
            if (ggml::maybe_inplace_op(tensor))
            {
                ggml::log(GGML_LOG_LEVEL_ERROR, "tensor %s might not work, op = %s", ggml::get_name(tensor), ggml::op_name(tensor));
            }
        }
    }

    ggml_backend_sched_t ComputeContext::get_sched(void)
    {
        return backend_context->sched.get();
    }

    void ComputeContext::move_to_layer(int layer_id)
    {
        backend_context->layer_allocators.move_to_layer(layer_id);
    }

    void ComputeContext::backend_cpu_override(bool flag)
    {
        backend_context->layer_allocators.override_to_cpu_only(flag);
    }

    BackendBufAllocator* ComputeContext::get_allocator(void)
    {
        return backend_context->layer_allocators.get_allocator();
    }

    BackendBufAllocator* ComputeContext::get_allocator(ggml::tensor* tensor)
    {
        return backend_context->layer_allocators.get_allocator(tensor);
    }

    void ComputeContext::register_tensor_allocator(ggml::tensor* tensor, BackendBufAllocator* allocator)
    {
        backend_context->layer_allocators.register_tensor_allocator(tensor, dynamic_cast<LayerBufAllocator*>(allocator));
    }

    Backend* ComputeContext::get_backend(void)
    {
        return dynamic_cast<LayerBufAllocator*>(get_allocator())->get_backend();
    }

    void ComputeContext::compute(void)
    {
        backend_context->compute_graph(get_cgraph());
    }

    void ComputeContext::synchronize(void)
    {
        get_sched()->synchronize();
    }

    bool ComputeContext::allocate(void)
    {
        return backend_context->alloc_graph(get_cgraph());
    }

    bool ComputeContext::reserve_memory(void)
    {
        return backend_context->reserve_memory(get_cgraph());
    }

    void ComputeContext::reset(void)
    {
        backend_context->reset();
    }

    void ComputeContext::set_backend_context(BackendContext* backend_context)
    {
        this->backend_context = backend_context;
    }
}
