module;
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <ranges>
#include <vector>
#define AT_PRINTF(...)
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :ds;
import :tensor;

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
	assert(alignment && !(alignment & (alignment - 1))); // power of 2
	size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
	return offset + align;
}

static bool ggml_op_can_inplace(enum ggml_op op) {
	switch (op) {
	case GGML_OP_SCALE:
	case GGML_OP_DIAG_MASK_ZERO:
	case GGML_OP_DIAG_MASK_INF:
	case GGML_OP_ADD:
	case GGML_OP_ADD_ID:
	case GGML_OP_ADD1:
	case GGML_OP_SUB:
	case GGML_OP_MUL:
	case GGML_OP_DIV:
	case GGML_OP_SQR:
	case GGML_OP_SQRT:
	case GGML_OP_LOG:
	case GGML_OP_UNARY:
	case GGML_OP_ROPE:
	case GGML_OP_RMS_NORM:
	case GGML_OP_SOFT_MAX:
		return true;

	default:
		return false;
	}
}

static bool ggml_is_view(const ggml_tensor* t) {
	return t->view_src != nullptr;
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
void ggml_dyn_tallocr::free_tensor(size_t offset, size_t size, const ggml_tensor* tensor) {
	size = aligned_offset(nullptr, size, alignment);

	AT_PRINTF("%s: freeing %s at %zu (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, offset, size, alloc->n_free_blocks);

#ifdef GGML_ALLOCATOR_DEBUG
	remove_allocated_tensor(this, offset, tensor);
#endif

	// see if we can merge with an existing block
	for (int i = 0; i < n_free_blocks; i++) {
		struct free_block* block = &free_blocks[i];
		// check if ptr is at the end of the block
		if (block->offset + block->size == offset) {
			block->size += size;
			// check if we can merge with the next block
			if (i < n_free_blocks - 1 && block->offset + block->size == free_blocks[i + 1].offset) {
				block->size += free_blocks[i + 1].size;
				n_free_blocks--;
				for (int j = i + 1; j < n_free_blocks; j++) {
					free_blocks[j] = free_blocks[j + 1];
				}
			}
			return;
		}
		// check if ptr is at the beginning of the block
		if (offset + size == block->offset) {
			block->offset = offset;
			block->size += size;
			// check if we can merge with the previous block
			if (i > 0 && free_blocks[i - 1].offset + free_blocks[i - 1].size == block->offset) {
				free_blocks[i - 1].size += block->size;
				n_free_blocks--;
				for (int j = i; j < n_free_blocks; j++) {
					free_blocks[j] = free_blocks[j + 1];
				}
			}
			return;
		}
	}
	// otherwise, add a new block
	GGML_ASSERT(n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
	// insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
	int insert_pos = 0;
	while (insert_pos < n_free_blocks && free_blocks[insert_pos].offset < offset) {
		insert_pos++;
	}
	// shift all blocks from insert_pos onward to make room for the new block
	for (int i = n_free_blocks; i > insert_pos; i--) {
		free_blocks[i] = free_blocks[i - 1];
	}
	// insert the new block
	free_blocks[insert_pos].offset = offset;
	free_blocks[insert_pos].size = size;
	n_free_blocks++;
}

size_t ggml_dyn_tallocr::alloc(size_t size, const ggml_tensor* tensor) {
	size = aligned_offset(NULL, size, alignment);

	AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

	size_t max_avail = 0;

	// find the best fitting free block besides the last block
	int best_fit_block = -1;
	size_t best_fit_size = SIZE_MAX;
	for (int i = 0; i < n_free_blocks - 1; i++) {
		struct free_block* block = &free_blocks[i];
		max_avail = std::max(max_avail, block->size);
		if (block->size >= size && block->size <= best_fit_size) {
			best_fit_block = i;
			best_fit_size = block->size;
		}
	}

	if (best_fit_block == -1) {
		// the last block is our last resort
		struct free_block* block = &free_blocks[n_free_blocks - 1];
		max_avail = std::max(max_avail, block->size);
		if (block->size >= size) {
			best_fit_block = n_free_blocks - 1;
		}
		else {
			// this should never happen
			GGML_LOG_ERROR("{}: not enough space in the buffer to allocate {} bytes, largest block available {} bytes",
				__func__, size, max_avail);
			GGML_ABORT("not enough space in the buffer");
		}
	}

	struct free_block* block = &free_blocks[best_fit_block];
	size_t offset = block->offset;
	block->offset = offset + size;
	block->size -= size;
	if (block->size == 0) {
		// remove block if empty
		n_free_blocks--;
		for (int j = best_fit_block; j < n_free_blocks; j++) {
			free_blocks[j] = free_blocks[j + 1];
		}
	}

	AT_PRINTF("block %d, offset %zu\n", best_fit_block, offset);

#ifdef GGML_ALLOCATOR_DEBUG
	add_allocated_tensor(alloc, offset, tensor);
	size_t cur_max = offset + size;
	if (cur_max > max_size) {
		// sort allocated_tensors by offset
		for (int i = 0; i < 1024; i++) {
			for (int j = i + 1; j < 1024; j++) {
				if (allocated_tensors[i].offset > allocated_tensors[j].offset) {
					const struct ggml_tensor* tmp_tensor = allocated_tensors[i].tensor;
					size_t tmp_offset = allocated_tensors[i].offset;
					allocated_tensors[i].tensor = allocated_tensors[j].tensor;
					allocated_tensors[i].offset = allocated_tensors[j].offset;
					allocated_tensors[j].tensor = tmp_tensor;
					allocated_tensors[j].offset = tmp_offset;
				}
			}
		}
		GGML_LOG_DEBUG("max_size = {:.2} MB: tensors: ", cur_max / 1024.0 / 1024.0);
		for (int i = 0; i < 1024; i++) {
			if (allocated_tensors[i].tensor) {
				GGML_LOG_DEBUG("{} [{:x}-{:x}] ({:.2} MB) ", allocated_tensors[i].tensor->name,
					allocated_tensors[i].offset,
					allocated_tensors[i].offset + ggml_nbytes(allocated_tensors[i].tensor),
					ggml_nbytes(allocated_tensors[i].tensor) / 1024.0 / 1024.0);
			}
		}
		GGML_LOG_DEBUG("\n");
	}
#endif

	max_size = std::max(max_size, offset + size);

	return offset;
}

bool ggml_gallocr::is_allocated(ggml_tensor* t) {
	return t->data != nullptr || hash_map[t].allocated;
}

bool ggml_gallocr::is_own(ggml_tensor* t) {
	return hash_map[t].allocated;
}

void ggml_gallocr::free_node(ggml_tensor* node) {
	// graph outputs are never freed
	if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
		AT_PRINTF("not freeing output %s\n", node->name);
		return;
	}

	auto& hn = hash_map[node];
	size_t offset = hn.offset;
	int buffer_id = hn.buffer_id;
	auto &alloc = buf_tallocs[buffer_id];
	ggml_backend_buffer_type_t buft = bufts[buffer_id];
	size_t size = buft->get_alloc_size(node);
	alloc->free_tensor(offset, size, node);
	hn.allocated = false;
}

void ggml_gallocr::allocate_node(ggml_tensor* node, int buffer_id) {
	GGML_ASSERT(buffer_id >= 0);
	if (!is_allocated(node) && !ggml_is_view(node)) {
		auto& hn = hash_map[node];
		hn.allocated = true;
		assert(hn.offset == 0);

		// try to reuse a parent's buffer (inplace)
		if (ggml_op_can_inplace(node->op)) {
			for (auto& parent : node->src) {
				// if the node's data is external, then we cannot re-use it
				if (!is_own(parent)) {
					AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
					continue;
				}

				// outputs cannot be reused
				if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
					AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
					continue;
				}

				if (!ggml_are_same_layout(node, parent)) {
					AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
					continue;
				}

				auto& p_hn = hash_map[parent];
				if (p_hn.n_children == 1 && p_hn.n_views == 0) {
					if (ggml_is_view(parent)) {
						struct ggml_tensor* view_src = parent->view_src;
						auto& view_src_hn = hash_map[view_src];
						if (view_src_hn.n_views == 1 && view_src_hn.n_children == 0 && view_src->data == parent->data) {
							AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
							assert(view_src_hn.offset == p_hn.offset);
							hn.buffer_id = p_hn.buffer_id;
							hn.offset = p_hn.offset;
							p_hn.allocated = false; // avoid freeing the parent
							view_src_hn.allocated = false;
							return;
						}
					}
					else {
						AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
						hn.buffer_id = p_hn.buffer_id;
						hn.offset = p_hn.offset;
						p_hn.allocated = false; // avoid freeing the parent
						return;
					}
				}
			}
		}

		// allocate tensor from the buffer
		auto& alloc = buf_tallocs[buffer_id];
		ggml_backend_buffer_type_t buft = bufts[buffer_id];
		size_t size = buft->get_alloc_size(node);
		size_t offset = alloc->alloc(size, node);
		hn.buffer_id = buffer_id;
		hn.offset = offset;
	}
}

void ggml_gallocr::alloc_graph_impl(const ggml_cgraph &graph,
	std::span<const int> node_buffer_ids,
	std::span<const int> leaf_buffer_ids) {
	hash_map.clear();
	// allocate leafs
	// these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
	for (size_t i = 0; i < graph.leafs.size(); i++) {
		auto& leaf = graph.leafs[i];
		allocate_node(leaf, leaf_buffer_ids[i]);
	}

	// count number of children and views
	// allocate other graph inputs and leafs first to avoid overwriting them
	for (size_t i = 0; i < graph.nodes.size(); i++) {
		auto& node = graph.nodes[i];
		// TODO: better way to add external dependencies
		// GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
		// control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
		// itself is never used and should not be considered a dependency
		if (ggml_is_view(node) && node->op != GGML_OP_NONE) {
			ggml_tensor* view_src = node->view_src;
			hash_map[view_src].n_views++;
		}

		if (node->flags & GGML_TENSOR_FLAG_INPUT) {
			allocate_node(node, node_buffer_ids[i]);
		}

		for (auto& src : node->src) {
			if (!src) continue;
			hash_map[src].n_children++;

			// allocate explicit inputs
			if (src->flags & GGML_TENSOR_FLAG_INPUT) {
				allocate_node(src, node_buffer_ids[i]);
			}
		}
	}

	// allocate tensors
	for (size_t i = 0; i < graph.nodes.size(); i++) {
		auto& node = graph.nodes[i];
		int buffer_id = node_buffer_ids[i];

		// allocate parents (only leafs need to be allocated at this point)
		for (auto& parent : node->src) {
			if (!parent) continue;
			allocate_node(parent, buffer_id);
		}

		// allocate node
		allocate_node(node, buffer_id);

		AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
		for (size_t j = 0; j < node->src.size(); j++) {
			auto& parent = node->src[j];
			AT_PRINTF("%s", parent->name);
			if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
				AT_PRINTF(", ");
			}
		}
		AT_PRINTF("\n");

		// update parents
		for (auto& parent : node->src) {
			auto& p_hn = hash_map[parent];
			p_hn.n_children -= 1;

			AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
				parent->name, p_hn.n_children, p_hn.n_views, p_hn.allocated);

			if (p_hn.n_children == 0 && p_hn.n_views == 0) {
				if (ggml_is_view(parent)) {
					struct ggml_tensor* view_src = parent->view_src;
					auto& view_src_hn = hash_map[view_src];
					view_src_hn.n_views -= 1;
					AT_PRINTF("view_src %s: %d children, %d views\n",
						view_src.name, view_src_hn.n_children, view_src_hn.n_views);
					if (view_src_hn.n_views == 0 && view_src_hn.n_children == 0 && view_src_hn.allocated) {
						free_node(view_src);
					}
				}
				else if (p_hn.allocated) {
					free_node(parent);
				}
			}
			AT_PRINTF("\n");
		}
	}
}

size_t ggml_gallocr::get_buffer_size(int buffer_id) {
	GGML_ASSERT(buffer_id >= 0 && buffer_id < buffers.size());

	if (!buffers[buffer_id]) {
		return 0;
	}

	for (int i = 0; i < buffer_id; i++) {
		if (buffers[i].get() == buffers[buffer_id].get()) {
			// this buffer is the same as a previous one due to the same buffer type being used multiple times
			// only return the buffer size the first time it appears to avoid double counting
			return 0;
		}
	}

	return buffers[buffer_id]->get_size();
}

void ggml_gallocr::init_tensor(ggml_tensor* tensor, tensor_alloc* tensor_alloc) {
	int buffer_id = tensor_alloc->buffer_id;
	assert(tensor->data || tensor->view_src || buffers[buffer_id]->get_alloc_size(tensor) <= tensor_alloc->size_max);

	if (tensor->view_src != NULL) {
		if (tensor->buffer == NULL) {
			assert(tensor_alloc->offset == SIZE_MAX);
			if (tensor->view_src->buffer == NULL) {
				// this tensor was allocated without ggml-backend
				return;
			}
			ggml_backend_view_init(tensor);
		}
	}
	else {
		if (tensor->data == NULL) {
			assert(tensor_alloc->offset != SIZE_MAX);
			assert(buffers[buffer_id]->get_alloc_size(tensor) <= tensor_alloc->size_max);
			void* base = buffers[buffer_id]->get_base();
			void* addr = (char*)base + tensor_alloc->offset;
			buffers[buffer_id]->alloc(tensor, addr);
		}
		else {
			if (tensor->buffer == NULL) {
				// this tensor was allocated without ggml-backend
				return;
			}
		}
	}
}

bool ggml_gallocr::node_needs_realloc(ggml_tensor* node, tensor_alloc* talloc) {
	size_t node_size = 0;
	if (!node->data && !node->view_src) {
		GGML_ASSERT(talloc->buffer_id >= 0); // prevent segfault when misusing the API
		node_size = bufts[talloc->buffer_id]->get_alloc_size(node);
	}
	return talloc->size_max >= node_size;
}

bool ggml_gallocr::needs_realloc(const ggml_cgraph& graph) {
	if (node_allocs.size() != graph.nodes.size()) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("{}: graph has different number of nodes", __func__);
#endif
		return true;
	}

	if (leaf_allocs.size() != graph.leafs.size()) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("{}: graph has different number of leafs", __func__);
#endif
		return true;
	}

	for (size_t i = 0; i < graph.nodes.size(); i++) {
		struct ggml_tensor* node = graph.nodes[i];
		struct node_alloc* node_alloc = &node_allocs[i];

		if (!node_needs_realloc(node, &node_alloc->dst)) {
#ifndef NDEBUG
			GGML_LOG_DEBUG("{}: node {} is not valid\n", __func__, node->name);
#endif
			return true;
		}

		for (size_t j = 0; j < node->src.size(); j++) {
			auto src = node->src[j];
			auto src_node_alloc = node_alloc->src[j];
			if (!src) continue;
			if (!node_needs_realloc(src, &src_node_alloc)) {
#ifndef NDEBUG
				GGML_LOG_DEBUG("{}: src {} ({}) of node {} is not valid", __func__, j, src->name, node->name);
#endif
				return true;
			}
		}
	}

	return false;
}

bool ggml_gallocr::alloc_graph(ggml_cgraph* graph) {
	if (needs_realloc(*graph)) {
		if (buffers.size() == 1) {
#ifndef NDEBUG
			GGML_LOG_DEBUG("{}: reallocating buffers automatically", __func__);
#endif
			if (!reserve(graph)) {
				return false;
			}
		}
		else {
#ifndef NDEBUG
			GGML_LOG_DEBUG("{}: cannot reallocate multi buffer graph automatically, call reserve", __func__);
#endif
			return false;
		}
	}

	// reset buffers
	for (auto& buffer : buffers) {
		if (buffer) {
			buffer->reset();
		}
	}

	// allocate the graph tensors from the previous assignments
	// leafs
	for (size_t i = 0; i < graph->leafs.size(); i++) {
		struct ggml_tensor* leaf = graph->leafs[i];
		struct leaf_alloc* leaf_alloc = &leaf_allocs[i];
		init_tensor(leaf, &leaf_alloc->leaf);
	}

	// nodes
	for (size_t i = 0; i < graph->nodes.size(); i++) {
		struct ggml_tensor* node = graph->nodes[i];
		struct node_alloc* node_alloc = &node_allocs[i];
		for (auto [src, node_alloc] : std::views::zip(node->src, node_alloc->src)) {
			if (!src) continue;
			init_tensor(src, &node_alloc);
		}
		init_tensor(node, &node_alloc->dst);
	}

	return true;
}

bool ggml_gallocr::reserve(const ggml_cgraph& graph, std::span<const int> node_buffer_ids, std::span<const int> leaf_buffer_ids) {
	// reset allocators
	for (auto& talloc : buf_tallocs) {
		talloc->reset();
	}

	// allocate in hash table
	alloc_graph_impl(graph, node_buffer_ids, leaf_buffer_ids);

	// set the node_allocs from the hash table
	node_allocs.resize(graph.nodes.size());

	for (size_t i = 0; i < graph.nodes.size(); i++) {
		auto &node = graph.nodes[i];
		auto &node_alloc = node_allocs[i];
		if (node->view_src || node->data) {
			node_alloc.dst.buffer_id = -1;
			node_alloc.dst.offset = SIZE_MAX;
			node_alloc.dst.size_max = 0;
		}
		else {
			const auto& hn = hash_map[node];
			node_alloc.dst.buffer_id = hn.buffer_id;
			node_alloc.dst.offset = hn.offset;
			node_alloc.dst.size_max = bufts[hn.buffer_id]->get_alloc_size(node);
		}
		for (size_t i = 0; i < node->src.size(); i++) {
			auto &src = node->src[i];
			auto &src_node_alloc = node_alloc.src[i];
			if (!src) continue;
			if (src->view_src || src->data) {
				src_node_alloc.buffer_id = -1;
				src_node_alloc.offset = SIZE_MAX;
				src_node_alloc.size_max = 0;
			}
			else {
				const auto& hn = hash_map[src];
				src_node_alloc.buffer_id = hn.buffer_id;
				src_node_alloc.offset = hn.offset;
				src_node_alloc.size_max = bufts[hn.buffer_id]->get_alloc_size(src);
			}
		}
	}

	leaf_allocs.resize(graph.leafs.size());
	for (size_t i = 0; i < graph.leafs.size(); i++) {
		auto &leaf = graph.leafs[i];
		auto &leaf_alloc = leaf_allocs[i];
		const auto& hn = hash_map[leaf];
		if (leaf->view_src || leaf->data) {
			leaf_alloc.leaf.buffer_id = -1;
			leaf_alloc.leaf.offset = SIZE_MAX;
			leaf_alloc.leaf.size_max = 0;
		}
		else {
			leaf_alloc.leaf.buffer_id = hn.buffer_id;
			leaf_alloc.leaf.offset = hn.offset;
			leaf_alloc.leaf.size_max = bufts[hn.buffer_id]->get_alloc_size(leaf);
		}
	}

	// reallocate buffers if needed
	for (size_t i = 0; i < buffers.size(); i++) {
		// if the buffer type is used multiple times, we reuse the same buffer
		for (size_t j = 0; j < i; j++) {
			if (buf_tallocs[j].get() == buf_tallocs[i].get()) {
				buffers[i] = buffers[j];
				break;
			}
		}

		size_t cur_size = buffers[i] ? buffers[i]->get_size() : 0;
		size_t new_size = buf_tallocs[i]->get_max_size();

		// even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
		if (new_size > cur_size || !buffers[i]) {
#ifndef NDEBUG
			GGML_LOG_DEBUG("{}: reallocating {} buffer from size {:.02} MiB to {:.02} MiB", __func__, bufts[i]->get_name(), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
			buffers[i] = bufts[i]->alloc_buffer(new_size);
			if (!buffers[i]) {
				GGML_LOG_ERROR("{}: failed to allocate {} buffer of size {}", __func__, bufts[i]->get_name(), new_size);
				return false;
			}
			buffers[i]->setUsage(GGML_BACKEND_BUFFER_USAGE_COMPUTE);
		}
	}

	return true;
}

bool ggml_gallocr::reserve(const ggml_cgraph* graph) {
	std::vector<int> nodes_zero(graph->nodes.size()), leafs_zero(graph->leafs.size());
	return reserve(*graph, nodes_zero, leafs_zero);
}

void ggml_tallocr::alloc(ggml_tensor* tensor) {
	size_t size = buffer->get_alloc_size(tensor);
	size = GGML_PAD(size, alignment);

	if (offset + size > buffer->get_size()) {
		GGML_LOG_ERROR("{}: not enough space in the buffer to allocate {} (needed {}, available {})",
			__func__, tensor->name, size, buffer->get_size() - offset);
		GGML_ABORT("not enough space in the buffer");
	}

	void* addr = (char*)buffer->get_base() + offset;
	offset += size;

	assert(((uintptr_t)addr % alignment) == 0);

	buffer->alloc(tensor, addr);
}

ggml_tallocr::ggml_tallocr(ggml_backend_buffer_t buffer) : buffer(buffer)
{
	base = buffer->get_base();
	alignment = buffer->get_alignment();
	assert(alignment && !(alignment & (alignment - 1))); // power of 2
	offset = aligned_offset(base, 0, alignment);
}
