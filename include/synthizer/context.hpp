#pragma once

#include "synthizer/base_object.hpp"
#include "synthizer/config.hpp"
#include "synthizer/invokable.hpp"
#include "synthizer/panner_bank.hpp"
#include "synthizer/property_internals.hpp"
#include "synthizer/property_ring.hpp"
#include "synthizer/router.hpp"
#include "synthizer/spatialization_math.hpp"
#include "synthizer/types.hpp"

#include "concurrentqueue.h"

#include <atomic>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace synthizer {

class AudioOutput;
class CExposable;
class Source;
class GlobalEffect;

/*
 * Infrastructure for deletion.
 * */
template<std::size_t ALIGN>
void contextDeferredFreeCallback(void *p) {
	if (ALIGN <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
		::operator delete(p);
	} else {
		::operator delete(p, (std::align_val_t)ALIGN);
	}
}

template<typename T>
void deletionCallback(void *p) {
	T *y = (T*) p;
	y->~T();
	deferredFree(contextDeferredFreeCallback<alignof(T)>, (void *)y);
}

/*
 * The context is the main entrypoint to Synthizer, holding the device, etc.
 * 
 *This has a few responsibilities:
 * - Dispatch callables to a high priority multimedia thread, with tagged priorities.
 * - Hold and orchestrate the lifetime of an audio device.
 * - Hold library-global parameters such as the listener's position and orientation, and configurable default values for things such as the distance model, speed of sound, etc.
 * - Handle memory allocation and freeing as necessary.
 * 
 * Users of synthizer will typically make one context per audio device they wish to use.
 * 
 * Unless otherwise noted, the functions of this class should only be called from the context-managed thread. External callers can submit invokables to run code on that thread, but 
 * since this is audio generation, the context needs full control over the priority of commands.
 * 
 * Later, if necessary, we'll extend synthizer to use atomics for some properties.
 * */
class Context: public BaseObject, public DistanceParamsMixin, public std::enable_shared_from_this<Context> {
	public:

	Context();
	/*
	 * Initialization occurs in two phases. The constructor does almost nothing, then this is called.
	 * 
	 * Why is because it is unfortunately necessary for the audio thread from miniaudio to hold a weak_ptr, which needs us to be able to use shared_from_this.
	 * */
	void initContext(bool headless = false);

	~Context();

	std::shared_ptr<Context> getContext() override;
	Context *getContextRaw() override {
		return this;
	}


	/*
	 * Shut the context down.
	 * 
	 * This kills the audio thread.
	 * */
	void shutdown();
	void cDelete() override;

	/*
	 * Submit an invokable which will be invoked on the context thread.
	 * */
	void enqueueInvokable(Invokable *invokable);

	/*
	 * Call a callable in the audio thread.
	 * Convenience method to not have to make invokables everywhere.
	 * */
	template<typename C, typename... ARGS>
	auto call(C &&callable, ARGS&& ...args) {
		auto cb = [&]() {
			return callable(args...);
		};
		if (this->headless) {
			return cb();
		}
		auto invokable = WaitableInvokable(std::move(cb));
		this->enqueueInvokable(&invokable);
		return invokable.wait();
	}

	template<typename T, typename... ARGS>
	std::shared_ptr<T> createObject(ARGS&& ...args) {
		auto obj = new T(this->shared_from_this(), args...);
		auto ret = sharedPtrDeferred<T>(obj, [] (T *ptr) {
			auto ctx = ptr->getContextRaw();
			if (ctx->delete_directly.load(std::memory_order_relaxed) == 0) ctx->enqueueDeletionRecord(&deletionCallback<T>, (void *)ptr);
			else delete ptr;
		});

		/* Do the second phase of initialization. */
		this->call([&] () {
			obj->initInAudioThread();
		});
		return ret;
	}

	/*
	 * get the current time since context creation in blocks.
	 * 
	 * This is used for crossfading and other applications.
	 * */
	unsigned int getBlockTime() {
		return this->block_time;
	}


	/*
	 * Helpers for the C API. to get/set properties in the context's thread.
	 * These create and manage the invokables and can be called directly.
	 * 
	 * Eventually this will be extended to handle batched/deferred things as well.
	 * */
	int getIntProperty(std::shared_ptr<BaseObject> &obj, int property);
	void setIntProperty(std::shared_ptr<BaseObject> &obj, int property, int value);
	double getDoubleProperty(std::shared_ptr<BaseObject> &obj, int property);
	void setDoubleProperty(std::shared_ptr<BaseObject> &obj, int property, double value);
	std::shared_ptr<CExposable> getObjectProperty(std::shared_ptr<BaseObject> &obj, int property);
	void setObjectProperty(std::shared_ptr<BaseObject> &obj, int property, std::shared_ptr<CExposable> &object);
	std::array<double, 3> getDouble3Property(std::shared_ptr<BaseObject> &obj, int property);
	void setDouble3Property(std::shared_ptr<BaseObject> &obj, int property, std::array<double, 3> value);
	std::array<double, 6>  getDouble6Property(std::shared_ptr<BaseObject> &obj, int property);
	void setDouble6Property(std::shared_ptr<BaseObject> &obj, int property, std::array<double, 6> value);

	/*
	 * Ad a weak reference to the specified source.
	 * 
	 * Handles calling into the audio thread.
	 * */
	void registerSource(const std::shared_ptr<Source> &source);

	/*
	 * Add a weak reference to the specified global effect.
	 * 
	 * Handles calling into the audio thread.
	 * */
	void registerGlobalEffect(const std::shared_ptr<GlobalEffect> &effect);

	/*
	 * The properties for the listener.
	 * */
	std::array<double, 3> getPosition();
	void setPosition(std::array<double, 3> pos);
	std::array<double, 6> getOrientation();
	void setOrientation(std::array<double, 6> orientation);

	/* Helper methods used by various pieces of synthizer to grab global resources. */

	/* Get the direct buffer, which is where things write when they want to bypass panning. This includes effects and direct sources, etc.
	 * Inline because it's super inexpensive.
	 * */
	float *getDirectBuffer() {
		return &this->direct_buffer[0];
	}


	/* Allocate a panner lane intended to be used by a source. */
	std::shared_ptr<PannerLane> allocateSourcePannerLane(enum SYZ_PANNER_STRATEGY strategy);

	router::Router *getRouter() {
		return &this->router;
	}

	/*
	 * Generate a block of audio output for the specified number of channels.
	 * 
	 * The number of channels shouldn't change for the duration of this context in most circumstances.
	 * */
	void generateAudio(unsigned int channels, float *output);

	#include "synthizer/property_methods.hpp"
	private:
	bool headless = false;

	/*
	 * Flush all pending property writes.
	 * */
	void flushPropertyWrites();

	moodycamel::ConcurrentQueue<Invokable *> pending_invokables;
	std::atomic<int> running;
	std::atomic<int> in_audio_callback = 0;
	std::shared_ptr<AudioOutput> audio_output;

	/*
	 * Deletion. This queue is read from when the semaphore for the context is incremented.
	 * 
	 * Objects are safe to delete when the iteration of the context at which the deletion was enqueued is greater.
	 * This means that all shared_ptr decremented in the previous iteration, and all weak_ptr were invalidated.
	 * */
	typedef void (*DeletionCallback)(void *);
	class DeletionRecord {
		public:
		uint64_t iteration;
		DeletionCallback callback;
		void *arg;
	};
	moodycamel::ConcurrentQueue<DeletionRecord> pending_deletes;

	std::atomic<int> delete_directly = 0;
	/*
	 * Used to signal that something is queueing a delete. This allows
	 * shutdown to synchronize by spin waiting, so that when it goes to drain the deletion queue, it can know that nothing else will appear in it.
	 * */
	std::atomic<int> deletes_in_progress = 0;

	void enqueueDeletionRecord(DeletionCallback cb, void *arg);
	/* Used by shutdown and the destructor only. Not safe to call elsewhere. */
	void drainDeletionQueues();

	PropertyRing<1024> property_ring;
	template<typename T>
	void propertySetter(const std::shared_ptr<BaseObject> &obj, int property, T &value);

	unsigned int block_time = 0;

	/* Collections of objects that require execution: sources, etc. all go here eventually. */

	/* direct_buffer is a buffer to which we write when we want to output, bypassing panner banks. */
	alignas(config::ALIGNMENT) std::array<float, config::BLOCK_SIZE * config::MAX_CHANNELS> direct_buffer;

	/* The key is a raw pointer for easy lookup. */
	deferred_unordered_map<void *, std::weak_ptr<Source>> sources;
	std::shared_ptr<AbstractPannerBank> source_panners = nullptr;

	/* Effects to run. */
	deferred_vector<std::weak_ptr<GlobalEffect>> global_effects;

	/* Parameters of the 3D environment: listener orientation/position, library-wide defaults for distance models, etc. */
	std::array<double, 3> position{ { 0, 0, 0 } };
	/* Default to facing positive y with positive x as east and positive z as up. */
	std::array<double, 6> orientation{ { 0, 1, 0, 0, 0, 1 } };

	/* Effects support. */
	router::Router router{};
};

}