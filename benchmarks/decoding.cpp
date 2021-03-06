#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include <cstddef>
#include <cmath>

#include "synthizer.h"
#include "synthizer_constants.h"

#define PI 3.1415926535
#define CHECKED(x) do { \
auto ret = x; \
	if (ret) { \
		printf(#x ": Synthizer error code %i message %s\n", ret, syz_getLastErrorMessage());\
		ecode = 1; \
		if (ending == 0) goto end; \
		else goto end2; \
	} \
} while(0)

int main(int argc, char *argv[]) {
	syz_Handle context = 0, buffer = 0;
	int ecode = 0, ending = 0;
	unsigned int iterations = 10;
	std::chrono::high_resolution_clock::time_point t_start, t_end;
	std::chrono::high_resolution_clock clock;
	unsigned int total_frames = 0, frames_tmp;

	if (argc != 2) {
		printf("Specify file to decode\n");
	}


	CHECKED(syz_configureLoggingBackend(SYZ_LOGGING_BACKEND_STDERR, nullptr));
	syz_setLogLevel(SYZ_LOG_LEVEL_DEBUG);
	CHECKED(syz_initialize());

	CHECKED(syz_createContext(&context));

	t_start = clock.now();
	for(unsigned int i = 0; i < iterations; i++) {
		CHECKED(syz_createBufferFromStream(&buffer, "file", argv[1], ""));
		CHECKED(syz_bufferGetLengthInSamples(&frames_tmp, buffer));
		total_frames += frames_tmp;
		CHECKED(syz_handleFree(buffer));
		/* if we fail to create the new one, let's no-op the free at the bottom. */
		buffer = 0;
	}
	t_end = clock.now();

	{
		auto dur = t_end - t_start;
		auto tmp = std::chrono::duration<double>(dur);
		double secs = tmp.count();
		printf("Took %f seconds total\n", secs);
		printf("%f per decode\n", secs / iterations);
		printf("Frames per second: %f\n", total_frames / secs);
	}

end:
	ending = 1;
	CHECKED(syz_handleFree(buffer));
	CHECKED(syz_handleFree(context));
end2:
	CHECKED(syz_shutdown());
	return ecode;
}
