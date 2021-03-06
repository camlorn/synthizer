#pragma once

#ifdef __cplusplus
extern "C" {
#endif 

enum SYZ_OBJECT_TYPE {
	SYZ_OTYPE_CONTEXT,
	SYZ_OTYPE_BUFFER,
	SYZ_OTYPE_BUFFER_GENERATOR,
	SYZ_OTYPE_STREAMING_GENERATOR,
	SYZ_OTYPE_NOISE_GENERATOR,
	SYZ_OTYPE_DIRECT_SOURCE,
	SYZ_OTYPE_PANNED_SOURCE,
	SYZ_OTYPE_SOURCE_3D,
	SYZ_OTYPE_GLOBAL_ECHO,
	SYZ_OTYPE_GLOBAL_FDN_REVERB,
};

enum SYZ_PANNER_STRATEGY {
	SYZ_PANNER_STRATEGY_HRTF = 0,
	SYZ_PANNER_STRATEGY_STEREO = 1,
	SYZ_PANNER_STRATEGY_COUNT,
};

/*
 * Distance models, modeled after the WebAudio spec.
 * */
enum SYZ_DISTANCE_MODEL {
	SYZ_DISTANCE_MODEL_NONE,
	SYZ_DISTANCE_MODEL_LINEAR,
	SYZ_DISTANCE_MODEL_EXPONENTIAL,
	SYZ_DISTANCE_MODEL_INVERSE,
	SYZ_DISTANCE_MODEL_COUNT,
};

enum SYZ_NOISE_TYPE {
	SYZ_NOISE_TYPE_UNIFORM,
	SYZ_NOISE_TYPE_VM,
	SYZ_NOISE_TYPE_FILTERED_BROWN,
	SYZ_NOISE_TYPE_COUNT,
};

enum SYZ_PROPERTIES {
	SYZ_P_AZIMUTH,
	SYZ_P_BUFFER,
	SYZ_P_CLOSENESS_BOOST,
	SYZ_P_CLOSENESS_BOOST_DISTANCE,
	SYZ_P_DISTANCE_MAX,
	SYZ_P_DISTANCE_MODEL,
	SYZ_P_DISTANCE_REF,
	SYZ_P_ELEVATION,
	SYZ_P_GAIN,
	SYZ_P_PANNER_STRATEGY,
	SYZ_P_PANNING_SCALAR,
	SYZ_P_POSITION,
	SYZ_P_ORIENTATION,
	SYZ_P_ROLLOFF,

	SYZ_P_LOOPING,

	SYZ_P_NOISE_TYPE,

	SYZ_P_PITCH_BEND,

	SYZ_P_INPUT_FILTER_ENABLED,
	SYZ_P_INPUT_FILTER_CUTOFF,
	SYZ_P_MEAN_FREE_PATH,
	SYZ_P_T60,
	SYZ_P_LATE_REFLECTIONS_LF_ROLLOFF,
	SYZ_P_LATE_REFLECTIONS_LF_REFERENCE,
	SYZ_P_LATE_REFLECTIONS_HF_ROLLOFF,
	SYZ_P_LATE_REFLECTIONS_HF_REFERENCE,
	SYZ_P_LATE_REFLECTIONS_DIFFUSION,
	SYZ_P_LATE_REFLECTIONS_MODULATION_DEPTH,
	SYZ_P_LATE_REFLECTIONS_MODULATION_FREQUENCY,
	SYZ_P_LATE_REFLECTIONS_DELAY,
};

enum SYZ_EVENT_TYPES {
	/* Invalid must always be 0. */
	SYZ_EVENT_TYPE_INVALID,
	SYZ_EVENT_TYPE_LOOPED,
	SYZ_EVENT_TYPE_FINISHED,
};

#ifdef __cplusplus
}
#endif
