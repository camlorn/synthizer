#pragma once

#include <cstdlib>

namespace synthizer {

/* Usage, from inside the synthizer namespace, is config::THING. */
namespace config {

/*
 * Sample rate of the library.
 * 
 * In order to be maximally efficient, both this and BLOCK_SIZE are fixed at compile-time.
 * */
const int SR = 44100;

/*
 * Number of samples to process in one block. This should be a multiple of 16 for future-proofing.
 * 
 * 256 samples is 172updatesper second, fast enough for anything reasonable.
 * 
 * We may have to raise this later for performance.
 * */
const int BLOCK_SIZE = 256;

/*
 * The maximum number of channels that any piece of Synthizer architecture can ever output.
 * 
 * This is used to allocate a number of buffers as static thread locals and/or on the stack, as opposed to on the heap.
 * */
const int MAX_CHANNELS = 16;

/*
 * When doing various internal crossfades (i.e. HRTF), how many samples do we use?
 * 
 * Must be a multiple of 4 and less than the block size; ideally keep this as a mutliple of 8.
 * */
const int CROSSFADE_SAMPLES = 64;

/*
 * The fundamental alignment, in bytes, of arrays holding samples.
 * 
 * SSE2 requires 16 byte alignment. Note that a float is 4 bytes, so we don't waste much.
 * */
const int ALIGNMENT = 16;

/*
 * The maximum delay for the ITD, in samples.
 * 
 * Must be at least 2.
 * 
 * This default comes from the woodworth ITD formula's maximum value for a 0.15 CM radius: (0.15/343)*(math.pi/2+1)*44100
 * Rounded up to a power of 2.
 * */
const int HRTF_MAX_ITD = 64;

/*
 * The maximum number of lanes a panner can ever have.
 * */
const int PANNER_MAX_LANES = 4;

/*
 * When storing buffers, how big should each page be? See buffer.hpp for explanation of how buffers work.
 * 
 * Should be a multiple of ALIGNMENT, but power of 2 is best.
 * 
 * Note: the primary trade-off here isn't memory fragmentation, it's speed at the boundaries.
 * */
const std::size_t BUFFER_CHUNK_SIZE = (1 << 14);

/*
 * maximum size of a command.
 * 
 * This is used to make the MpscRing entirely inline by using aligned_storage.
 * */
const std::size_t MAX_COMMAND_SIZE = 128;

}
}