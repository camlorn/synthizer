#include "synthizer/biquad.hpp"

#include "synthizer/config.hpp"
#include "synthizer/filter_design.hpp"
#include "synthizer/iir_filter.hpp"
#include "synthizer/memory.hpp"

#include <array>
#include <memory>

namespace synthizer {

template<unsigned int CHANNELS>
class ConcreteBiquadFilter: public BiquadFilter {
	public:
	ConcreteBiquadFilter();

	void processBlock(float *in, float *out, bool add = true) override;
	void configure(const syz_BiquadConfig &config) override;

	private:
	template<bool ADD, bool CROSSFADING>
	void processBlockImpl(float *in, float *out);
	std::array<IIRFilter<CHANNELS, 3, 3>, 2> filters;
	bool crossfade = false;
	/* Points to the index of the active filter in the array. */
	unsigned char active = 0;
};

template<unsigned int CHANNELS>
ConcreteBiquadFilter<CHANNELS>::ConcreteBiquadFilter() {
	this->filters[0].setParameters(designWire());
	this->filters[1].setParameters(designWire());
}

template<unsigned int CHANNELS>
void ConcreteBiquadFilter<CHANNELS>::configure(const syz_BiquadConfig &config) {
	BiquadFilterDef def;
	def.num_coefs[0] = config.b0;
	def.num_coefs[1] = config.b1;
	def.num_coefs[2] = config.b2;
	def.den_coefs[0] = config.a1;
	def.den_coefs[1] = config.a2;
	def.gain = config.gain;
	this->filters[this->active ^ 1].setParameters(def);
	this->crossfade = true;
}

template<unsigned int CHANNELS>
void ConcreteBiquadFilter<CHANNELS>::processBlock(float *in, float *out, bool add) {
	if (add) {
		if (this->crossfade) {
			return this->processBlockImpl<true, true>(in, out);
		} else {
			return this->processBlockImpl<true, false>(in, out);
		}
	} else {
		if (this->crossfade) {
			return this->processBlockImpl<false, true>(in, out);
		} else {
			return this->processBlockImpl<false, true>(in, out);
		}
	}

	if (this->crossfade) {
		/*
		 * We only ever crossfade for 1 block, after reconfiguring. Stop crossfading and flip
		 * the active filter.
		 * */
		this->crossfade = false;
		this->active ^= 1;
	}
}

template<unsigned int CHANNELS>
template<bool ADD, bool CROSSFADE>
void ConcreteBiquadFilter<CHANNELS>::processBlockImpl(float *in, float *out) {
	const float gain_inv = 1.0f / config::BLOCK_SIZE;
	IIRFilter<CHANNELS, 3, 3> *active = &this->filters[this->active];
	IIRFilter<CHANNELS, 3, 3> *inactive = &this->filters[this->active ^ 1];

	for (unsigned int i = 0; i < config::BLOCK_SIZE; i++) {
		float tmp[CHANNELS] = { 0.0f };
		float *in_frame = in + CHANNELS * i;
		/*
		 * Unfortunately the IIRFilter always sets, because it's old enough
		 * to be from before Synthizer established firm conventions.
		 * */
		float *out_frame = ADD || CROSSFADE ? out + CHANNELS * i : &tmp[0];

		active->tick(in_frame, out_frame);
		if (CROSSFADE) {
			/*
			 * Fade inactive out, fade active in.
			 * */
			float cf_tmp[CHANNELS] = { 0.0f };
			inactive->tick(in_frame, cf_tmp);
			float w2 = i * gain_inv;
			float w1 = 1.0f - w2;
			for (unsigned int c = 0; c < CHANNELS; c++) {
				tmp[c] = cf_tmp[c] * w1 + tmp[c] * w2;
			}
		}

		if (CROSSFADE || ADD) {
			/* We used the intermediate buffer, so copy out. */
			for (unsigned int c = 0; c < CHANNELS; c++) {
				float res = ADD ? out_frame[c] + tmp[c] : tmp[c];
				out_frame[c] = res;
			}
		}
		/* If we didn't crossfade or add, it's already where it should be. */
	}
}

template<unsigned int CHANNELS>
static std::shared_ptr<BiquadFilter> biquadFilterFactory() {
	auto obj = new ConcreteBiquadFilter<CHANNELS>();
	auto ret = sharedPtrDeferred<ConcreteBiquadFilter<CHANNELS>>(obj, [](ConcreteBiquadFilter<CHANNELS> *o) {
		deferredDelete(o);
	});
	return std::static_pointer_cast<BiquadFilter>(ret);
}

/*
 * rather than use std::array here, we use boring C-like arrays. This is because
 * we want to infer then static_assert the length, which we can't do if we use std::array (which won't defer).
 * */
typedef std::shared_ptr<BiquadFilter> BiquadFilterFactoryCb();
static BiquadFilterFactoryCb *factories[] = {
	biquadFilterFactory<1>,
	biquadFilterFactory<2>,
	biquadFilterFactory<3>,
	biquadFilterFactory<4>,
	biquadFilterFactory<5>,
	biquadFilterFactory<6>,
	biquadFilterFactory<7>,
	biquadFilterFactory<8>,
	biquadFilterFactory<9>,
	biquadFilterFactory<10>,
	biquadFilterFactory<11>,
	biquadFilterFactory<12>,
	biquadFilterFactory<13>,
	biquadFilterFactory<14>,
	biquadFilterFactory<15>,
	biquadFilterFactory<16>,
};

static_assert(sizeof(factories) / sizeof(factories[0]) == config::MAX_CHANNELS, "Neeed to add/remove biquad factories if MAX_CHANNELS is changed");

std::shared_ptr<BiquadFilter> createBiquadFilter(unsigned int channels) {
	assert(channels > 0);
	assert(channels <= config::MAX_CHANNELS);
	return factories[channels - 1]();
}

}
