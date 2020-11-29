#pragma once

#include <array>
#include <limits>
#include <memory>
#include <variant>

namespace synthizer {

/*
 * Properties work through a DSL using X macros and a ton of magic. This file contains helpers for the magic.
 * 
 * To add a property, follow the instructions in synthizer_properties.h, then define the following macros and include property_impl.hpp (which can be done more than once per file).
 * 
 * - PROPERTY_CLASS: the class to implement property functionality for.
 * - PROPERTY_BASE: the base class one level up
 * - PROPPERTY_LIST: the macro from synthizer_properties.h that has the properties for this class.
 * 
 * For example:
 * 
 * #define PROPERTY_CLASS PannedSource
 * #define PROPERTY_BASE Source
 * #define PROPERTY_LIST PANNED_SOURCE_PROPERTIES
 * #include "synthizer/property_impl.hpp"
 * 
 * the effect is to override a number of methods from the base (ultimately BaseObject) to provide a getProperty and setProperty that use variants which back onto getXXX and setXXX functions.
 * */

class CExposable;

/* We hide this in a namespace because it's only for macro magic and shouldn't be used directly by anything else. */
namespace property_impl {

using PropertyValue = std::variant<int, double, std::shared_ptr<CExposable>, std::array<double, 3>, std::array<double, 6>>;

/* array<double, 3. doesn't work in macros because of the comma. */
using arrayd3 = std::array<double, 3>;
using arrayd6 = std::array<double, 6>;

const auto int_min = std::numeric_limits<int>::min();
const auto int_max = std::numeric_limits<int>::max();
const auto double_min = std::numeric_limits<double>::lowest();
const auto double_max = std::numeric_limits<double>::max();
}

}
