module;
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <memory>
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
	case GGML_OP_FILL:
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
void ggml_dyn_tallocr::free_bytes(buffer_address addr, size_t size) {
	size = aligned_offset(NULL, size, alignment);

	tallocr_chunk& chunk = chunks[addr.chunk];

	// see if we can merge with an existing block
	for (int i = 0; i < chunk.n_free_blocks; i++) {
		free_block* block = &chunk.free_blocks[i];
		// check if ptr is at the end of the block
		if (block->offset + block->size == addr.offset) {
			block->size += size;
			// check if we can merge with the next block
			if (i < chunk.n_free_blocks - 1) {
				struct free_block* next = &chunk.free_blocks[i + 1];
				if (block->offset + block->size == next->offset) {
					block->size += next->size;
					chunk.remove_block(i + 1);
				}
			}
			return;
		}
		// check if ptr is at the beginning of the block
		if (addr.offset + size == block->offset) {
			block->offset = addr.offset;
			block->size += size;
			// check if we can merge with the previous block
			if (i > 0) {
				free_block* prev = &chunk.free_blocks[i - 1];
				if (prev->offset + prev->size == block->offset) {
					prev->size += block->size;
					chunk.remove_block(i);
				}
			}
			return;
		}
	}
	// otherwise, add a new block
	chunk.insert_block(addr.offset, size);
}

tallocr_chunk* ggml_dyn_tallocr::new_chunk(size_t min_size) {
	if (chunks.size() >= GGML_VBUFFER_MAX_CHUNKS) {
		return nullptr;
	}

	tallocr_chunk& chunk = chunks.emplace_back();
	chunk.n_free_blocks = 1;
	chunk.free_blocks[0].offset = 0;
	// available space in a chunk is limited to max_chunk_size, but can be higher if:
	// 1. a single tensor exceeds the maximum, and cannot fit any other way
	// 2. we are running out of chunks
	// backends will either manage to allocate the larger size, or report an error.
	chunk.free_blocks[0].size = std::max(min_size, max_chunk_size);
	if (chunks.size() == GGML_VBUFFER_MAX_CHUNKS) {
		chunk.free_blocks[0].size = SIZE_MAX / 2;
	}
	return &chunk;
}

buffer_address ggml_dyn_tallocr::alloc(size_t size, const ggml_tensor* tensor) {
	size = aligned_offset(NULL, size, alignment);

	AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

	tallocr_chunk* best_fit_chunk = nullptr;
	int best_fit_chunk_index = -1;
	int best_fit_block = -1;
	size_t max_avail = 0;

	// find the best fitting free block besides the last block, within any chunk
	for (auto [index, chunk] : chunks | std::views::enumerate) {
		size_t best_fit_size = SIZE_MAX;
		for (int i = 0; i < chunk.n_free_blocks - 1; i++) {
			free_block* block = &chunk.free_blocks[i];
			max_avail = std::max(max_avail, block->size);
			if (block->size >= size && block->size <= best_fit_size) {
				best_fit_chunk = &chunk;
				best_fit_chunk_index = index;
				best_fit_block = i;
				best_fit_size = block->size;
			}
		}
	}
	if (best_fit_block == -1) {
		// no suitable block found, try the last block (this may grow a chunks size)
		int64_t best_reuse = INT64_MIN;
		for (auto [index, chunk] : chunks | std::views::enumerate) {
			if (chunk.n_free_blocks > 0) {
				free_block* block = &chunk.free_blocks[chunk.n_free_blocks - 1];
				max_avail = std::max(max_avail, block->size);
				int64_t reuse_factor = chunk.max_size - block->offset - size;
				// reuse_factor < 0 : amount of extra memory that needs to be allocated
				// reuse_factor = 0 : allocated free space exactly matches tensor size
				// reuse_factor > 0 : superfluous memory that will remain unused
				bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
				bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
				if (block->size >= size && (better_reuse || better_fit)) {
					best_fit_chunk = &chunk;
					best_fit_chunk_index = index;
					best_fit_block = chunk.n_free_blocks - 1;
					best_reuse = reuse_factor;
				}
			}
		}
	}

	if (best_fit_block == -1) {
		// none of the existing chunks have enough space left
		best_fit_chunk = new_chunk(size);
		best_fit_chunk_index = chunks.size() - 1;
		best_fit_block = 0;
	}
	if (best_fit_chunk == nullptr) {
		// since the last chunk always has virtually endless memory, this should never happen
		GGML_LOG_ERROR("{}: not enough space in the buffer to allocate {} bytes, largest block available {} bytes\n",
			__func__, size, max_avail);
		GGML_ABORT("graph allocation: failed to reserve memory");
	}

	free_block* block = &best_fit_chunk->free_blocks[best_fit_block];
	buffer_address  addr = { 
		.chunk = best_fit_chunk_index,
		.offset = block->offset 
	};
	block->offset += size;
	block->size -= size;
	if (block->size == 0) {
		// remove block if empty
		best_fit_chunk->remove_block(best_fit_block);
	}

	AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
	add_allocated_tensor(alloc, addr, tensor);
	size_t cur_max = addr.offset + size;
	if (cur_max > chunk->max_size) {
		// sort allocated_tensors by chunk/offset
		for (int i = 0; i < 1024; i++) {
			for (int j = i + 1; j < 1024; j++) {
				if (ggml_buffer_address_less(allocated_tensors[j].addr, allocated_tensors[i].addr)) {
					const struct ggml_tensor* tmp_tensor = allocated_tensors[i].tensor;
					struct buffer_address tmp_addr = allocated_tensors[i].addr;
					allocated_tensors[i].tensor = allocated_tensors[j].tensor;
					allocated_tensors[i].addr = allocated_tensors[j].addr;
					allocated_tensors[j].tensor = tmp_tensor;
					allocated_tensors[j].addr = tmp_addr;
				}
			}
		}
		GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
		for (int i = 0; i < 1024; i++) {
			if (allocated_tensors[i].tensor) {
				GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", allocated_tensors[i].tensor->name,
					allocated_tensors[i].addr.chunk,
					allocated_tensors[i].addr.offset,
					allocated_tensors[i].addr.offset + ggml_nbytes(allocated_tensors[i].tensor),
					ggml_nbytes(allocated_tensors[i].tensor) / 1024.0 / 1024.0);
			}
		}
		GGML_LOG_DEBUG("\n");
	}
#endif

	best_fit_chunk->max_size = std::max(best_fit_chunk->max_size, addr.offset + size);

	return addr;
}

bool ggml_gallocr::is_allocated(ggml_tensor* t) {
	return t->data != nullptr // tensor data already set externally
		|| t->buffer // tensor on external buffer (but not yet allocated)
		|| is_own(t); // tensor will be allocated by galloc
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
	int buffer_id = hn.buffer_id;
	auto& alloc = buf_tallocs[buffer_id];
	ggml_backend_buffer_type* buft = bufts[buffer_id];
	size_t size = buft->get_alloc_size(node);

	AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
		__func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
	remove_allocated_tensor(alloc, hn->addr, node);
#endif

	alloc->free_bytes(hn.addr, size);
	hn.allocated = false;
}

// free the extra space at the end if the new tensor is smaller
void ggml_gallocr::free_extra_space(ggml_tensor* node, ggml_tensor* parent) {
	auto& hn = hash_map[node];
	auto& p_hn = hash_map[parent];

	size_t parent_size = bufts[p_hn.buffer_id]->get_alloc_size(parent);
	size_t node_size = bufts[hn.buffer_id]->get_alloc_size(node);

	GGML_ASSERT(parent_size >= node_size);

	// note: we want after the freeing the chunks to continue to be aligned
	auto p_alloc = buf_tallocs[p_hn.buffer_id];
	parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
	node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

	if (parent_size > node_size) {
		buffer_address p_addr = p_hn.addr;
		p_addr.offset += node_size;
		size_t extra_size = parent_size - node_size;
		AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
		p_alloc->free_bytes(p_addr, extra_size);
	}
}

void ggml_gallocr::allocate_node(ggml_tensor* node, int buffer_id) {
	GGML_ASSERT(buffer_id >= 0);
	if (!is_allocated(node) && !ggml_is_view(node)) {
		auto& hn = hash_map[node];
		hn.allocated = true;
		assert(hn.addr.offset == 0);

		// try to reuse a parent's buffer (inplace)
		if (ggml_op_can_inplace(node->op)) {
			for (int i = 0; i < GGML_MAX_SRC; i++) {
				ggml_tensor* parent = node->src[i];
				if (parent == NULL) {
					continue;
				}

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

				auto &p_hn = hash_map[parent];
				if (p_hn.n_children == 1 && p_hn.n_views == 0) {
					if (ggml_is_view(parent)) {
						ggml_tensor* view_src = parent->view_src;
						auto &view_src_hn = hash_map[view_src];
						if (view_src_hn.n_views == 1 && view_src_hn.n_children == 0 && view_src->data == parent->data) {
							AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
							assert(view_src_hn.addr.chunk == p_hn.addr.chunk && view_src_hn.addr.offset == p_hn.addr.offset);
							hn.buffer_id = p_hn.buffer_id;
							hn.addr = p_hn.addr;
							p_hn.allocated = false; // avoid freeing the parent
							view_src_hn.allocated = false;
							free_extra_space(node, view_src);
							return;
						}
					}
					else {
						AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
						hn.buffer_id = p_hn.buffer_id;
						hn.addr = p_hn.addr;
						p_hn.allocated = false; // avoid freeing the parent
						free_extra_space(node, parent);
						return;
					}
				}
			}
		}
		// allocate tensor from the buffer
		auto& alloc = buf_tallocs[buffer_id];
		ggml_backend_buffer_type* buft = bufts[buffer_id];
		size_t size = buft->get_alloc_size(node);
		hn.buffer_id = buffer_id;
		hn.addr = alloc->alloc(size, node);
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

	return buffers[buffer_id]->size();
}

void ggml_gallocr::init_tensor(ggml_tensor* tensor, tensor_alloc* tensor_alloc) {
	int buffer_id = tensor_alloc->buffer_id;
	assert(tensor->data || tensor->view_src || bufts[buffer_id]->get_alloc_size(tensor) <= tensor_alloc->size_max);

	if (tensor->view_src != NULL) {
		if (tensor->buffer == NULL) {
			assert(tensor_alloc->addr.offset == SIZE_MAX);
			if (tensor->view_src->buffer == NULL) {
				// this tensor was allocated without ggml-backend
				return;
			}
			ggml_backend_view_init(tensor);
		}
	}
	else {
		if (tensor->data == NULL) {
			assert(tensor_alloc->addr.offset != SIZE_MAX);
			assert(bufts[buffer_id]->get_alloc_size(tensor) <= tensor_alloc->size_max);
			buffers[buffer_id]->alloc(tensor, tensor_alloc->addr);
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
			if (!reserve(*graph)) {
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

static const buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

bool ggml_gallocr::reserve(const ggml_cgraph& graph,
	std::span<const int> node_buffer_ids, std::span<const int> leaf_buffer_ids, bool no_alloc)
{
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
            node_alloc.dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc.dst.size_max = 0;
		}
		else {
			const auto& hn = hash_map[node];
			node_alloc.dst.buffer_id = hn.buffer_id;
			node_alloc.dst.addr = hn.addr;
			node_alloc.dst.size_max = bufts[hn.buffer_id]->get_alloc_size(node);
		}
		for (size_t i = 0; i < node->src.size(); i++) {
			auto &src = node->src[i];
			auto &src_node_alloc = node_alloc.src[i];
			if (!src) continue;
			if (src->view_src || src->data) {
				src_node_alloc.buffer_id = -1;
				src_node_alloc.addr = GGML_BUFFER_ADDRESS_INVALID;
				src_node_alloc.size_max = 0;
			}
			else {
				const auto& hn = hash_map[src];
				src_node_alloc.buffer_id = hn.buffer_id;
				src_node_alloc.addr = hn.addr;
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
			leaf_alloc.leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
			leaf_alloc.leaf.size_max = 0;
		}
		else {
			leaf_alloc.leaf.buffer_id = hn.buffer_id;
			leaf_alloc.leaf.addr = hn.addr;
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

		// even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
		bool realloc = buffers[i].get() == nullptr;
		size_t new_size = 0;
		for (int c = 0; c < buf_tallocs[i]->chunks.size(); c++) {
			size_t cur_chunk_size = buffers[i] ? buffers[i]->chunk_size(c) : 0;
			size_t new_chunk_size = buf_tallocs[i]->max_size(c);
			new_size += new_chunk_size;
			if (new_chunk_size > cur_chunk_size) {
				realloc = true;
			}
		}
		if (realloc) {
#ifndef NDEBUG
			{
				size_t cur_size = buffers[i] ? buffers[i]->size() : 0;
				if (cur_size > 0) {
					GGML_LOG_DEBUG("{}: reallocating {} buffer from size {:.02f} MiB to {:.02f} MiB\n",
						__func__, bufts[i]->get_name(), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
				}
			}
#endif
			if (no_alloc) {
				buffers[i].reset();
			}
			else {
				buffers[i] = std::make_shared<vbuffer>(bufts[i], buf_tallocs[i].get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
				if (buffers[i] == nullptr) {
					GGML_LOG_ERROR("{}: failed to allocate {} buffer of size %zu\n", __func__, bufts[i]->get_name(), new_size);
					return false;
				}
			}
		}
	}

	return true;
}

bool ggml_gallocr::reserve(const ggml_cgraph &graph) {
	std::vector<int> nodes_zero(graph.nodes.size()), leafs_zero(graph.leafs.size());
	return reserve(graph, nodes_zero, leafs_zero);
}

bool ggml_gallocr::reserve(const ggml_cgraph& graph,
	std::span<const int> node_buffer_ids, std::span<const int> leaf_buffer_ids)
{
	return reserve(graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

void ggml_gallocr::reserve_n_size(const ggml_cgraph &graph, std::span<const int> node_buffer_ids,
	std::span<const int> leaf_buffer_ids, size_t* sizes)
{
	assert(reserve(graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
	for (int i = 0; i < buffers.size(); i++) {
		sizes[i] = 0;
		for (auto &chunk : buf_tallocs[i]->chunks) {
			sizes[i] += chunk.max_size;
		}
	}
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

ggml_tallocr::ggml_tallocr(ggml_backend_buffer* buffer) : buffer(buffer)
{
	base = buffer->get_base();
	alignment = buffer->get_alignment();
	assert(alignment && !(alignment & (alignment - 1))); // power of 2
	offset = aligned_offset(base, 0, alignment);
}

vbuffer::vbuffer(ggml_backend_buffer_type* buft, const ggml_dyn_tallocr* talloc, ggml_backend_buffer_usage usage)
{
	for (int n = 0; n < talloc->chunks.size(); n++) {
		size_t chunk_size = talloc->chunks[n].max_size;
		auto &new_chunk = chunks.push_back(buft->alloc_buffer(chunk_size));
#if 0
		if (new_chunk == nullptr) {
			ggml_vbuffer_free(buf);
			return NULL;
		}
#endif
		new_chunk->setUsage(usage);
	}
}

size_t vbuffer::chunk_size(int chunk)
{
	return chunk < chunks.size() ? chunks[chunk]->get_size() : 0;
}

size_t vbuffer::size() const
{
	size_t size = 0;
	for (auto &chunk : chunks) size += chunk->get_size();
	return size;
}

void vbuffer::alloc(ggml_tensor* tensor, buffer_address buf_addr)
{
	void* base = chunks[buf_addr.chunk]->get_base();
	void* addr = (char*)base + buf_addr.offset;
	chunks[buf_addr.chunk]->alloc(tensor, addr);
}

void vbuffer::reset()
{
	for (auto& chunk : chunks) chunk->reset();
}

void tallocr_chunk::remove_block(int idx) {
	// shift all elements after idx by 1 to the left, overwriting the element at idx
	for (int i = idx; i < n_free_blocks; i++) {
		free_blocks[i] = free_blocks[i + 1];
	}
	n_free_blocks--;
}

void tallocr_chunk::insert_block(size_t offset, size_t size)
{
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