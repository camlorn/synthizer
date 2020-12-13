# C API Conventions

The headers are:

- `synthizer.h`: All library functions
- `synthizer_constants.h`: Constants, i.e. the very large property enum.

The Synthizer C API returns errors and writes results to out parameters.  Out parameters are always the first parameters of a function, and errors are always nonzero.  Note that error codes are currently not defined; they will be, once things are more stable.
