/*
 * Defines property methods on a class.
 * 
 * This header can be included more than once, and should be included outside the synthizer namespace.
 * */

#include "synthizer_constants.h"
#include "synthizer/property_xmacros.hpp"

#include "synthizer/base_object.hpp"
#include "synthizer/error.hpp"
#include "synthizer/memory.hpp"
#include "synthizer/property_internals.hpp"

#include <memory>
#include <variant>

namespace synthizer {

#ifndef PROPERTY_CLASS
#error "When implementing properties, must define PROPERTY_CLASS to the class the properties are being added to."
#endif

#ifndef PROPERTY_BASE
#error "Forgot to define PROPERTY_BASE when implementing properties. Define this to your immediate base class."
#endif

#ifndef PROPERTY_LIST
#error "Need PROPERTY_LIST defined to know where to get properties from"
#endif

#ifdef PROPERTY_CLASS_IS_TEMPLATE
#define TEMPLATE_HEADER template<>
#else
#define TEMPLATE_HEADER
#endif

#define P_INT_MIN property_impl::int_min
#define P_INT_MAX property_impl::int_max
#define P_DOUBLE_MIN property_impl::double_min
#define P_DOUBLE_MAX property_impl::double_max

/*
 * The implementation itself is a giant switch, with a call in the default case to the base for 3 methods.
 * 
 * These are the stanzas for those methods.
 * */
#define HAS_(P, ...) \
case (P): return true; \

#define GET_CONV_(t, conv, p, name1, name2, ...) \
case (p): { \
	auto tmp = this->get##name2(); \
	property_impl::PropertyValue ret = conv(tmp); \
	auto ptr = std::get_if<t>(&ret); \
	if (ptr == nullptr) throw EPropertyType(); \
	return ret; \
} \
break;

#define GET_(t, ...) GET_CONV_(t, [](auto x) { return x; }, __VA_ARGS__)

#define VALIDATE_(type, checker, p, ...) \
	case (p): { \
		auto ptr = std::get_if<type>(&value); \
		if (ptr == nullptr) throw EPropertyType(); \
		checker(ptr); \
		break; \
	}

#define SET_(type, conv, p, name1, name2, ...) \
case (p): { \
	auto ptr = std::get_if<type>(&value); \
	if (ptr == nullptr) throw EPropertyType(); \
	this->set##name2(conv(ptr)); \
} \
break;

/* Now implement the methods. */

#define INT_P(...) HAS_(__VA_ARGS__)
#define DOUBLE_P(...) HAS_(__VA_ARGS__)
#define OBJECT_P(...) HAS_(__VA_ARGS__)
#define DOUBLE3_P(...) HAS_(__VA_ARGS__)
#define DOUBLE6_P(...) HAS_(__VA_ARGS__)

TEMPLATE_HEADER
bool PROPERTY_CLASS::hasProperty(int property) {
	switch (property) {
	PROPERTY_LIST
	default: 
		PROPERTY_BASE::hasProperty(property);
	}

	return false;
}

#undef INT_P
#undef DOUBLE_P
#undef OBJECT_P
#undef DOUBLE3_P
#undef DOUBLE6_P

#define INT_P(...) GET_(int, __VA_ARGS__)
#define DOUBLE_P(...) GET_(double, __VA_ARGS__)
#define DOUBLE3_P(...) GET_(property_impl::arrayd3, __VA_ARGS__)
#define DOUBLE6_P(...) GET_(property_impl::arrayd6, __VA_ARGS__)

/* Getting objects is different; we have to cast to the base class. */
#define OBJECT_P(p, ...) GET_CONV_(std::shared_ptr<CExposable>, [] (auto &x) { return std::static_pointer_cast<CExposable>(x); }, p, __VA_ARGS__)

TEMPLATE_HEADER
property_impl::PropertyValue PROPERTY_CLASS::getProperty(int property) {
	switch (property) {
	PROPERTY_LIST
	default:
		return PROPERTY_BASE::getProperty(property);
	}
}

#undef INT_P
#undef DOUBLE_P
#undef OBJECT_P
#undef DOUBLE3_P
#undef DOUBLE6_P

#define INT_P(p, name1, name2, min, max, ...) VALIDATE_(int, [](auto x) { if(*x < min || *x > max) throw ERange(); }, p, name1, name2, min, max);
#define DOUBLE_P(p, name1, name2, min, max, ...) VALIDATE_(double, [](auto x) { if (*x < min || *x > max) throw ERange(); }, p, name1, name2, min, max)
#define OBJECT_P(p, name1, name2, cls) VALIDATE_(std::shared_ptr<CExposable>, [] (auto *x) { if (*x == nullptr) return; auto &y = *x; auto z = std::dynamic_pointer_cast<cls>(y); if (z == nullptr) throw EHandleType(); }, p, name1, name2, cls)
#define DOUBLE3_P(...)  VALIDATE_(property_impl::arrayd3, [] (auto &x) {}, __VA_ARGS__)
#define DOUBLE6_P(...)  VALIDATE_(property_impl::arrayd6, [] (auto &x) {}, __VA_ARGS__)

TEMPLATE_HEADER
void PROPERTY_CLASS::validateProperty(int property, const property_impl::PropertyValue &value) {
	switch (property) {
	PROPERTY_LIST
	default:
		PROPERTY_BASE::validateProperty(property, value);
	}
}

#undef INT_P
#undef DOUBLE_P
#undef OBJECT_P
#undef DOUBLE3_P
#undef DOUBLE6_P

#define INT_P(p, name1, name2, min, max, ...) SET_(int, [](auto x) { return *x; }, p, name1, name2, min, max);
#define DOUBLE_P(p, name1, name2, min, max, ...) SET_(double, [](auto x) { return *x; }, p, name1, name2, min, max)
#define OBJECT_P(p, name1, name2, cls) SET_(std::shared_ptr<CExposable>, [] (auto *x) -> std::shared_ptr<cls> { return *x ? std::static_pointer_cast<cls>(*x) : nullptr ; }, p, name1, name2, cls)
#define DOUBLE3_P(...)  SET_(property_impl::arrayd3, [] (auto &x) { return *x; }, __VA_ARGS__)
#define DOUBLE6_P(...)  SET_(property_impl::arrayd6, [] (auto &x) { return *x; }, __VA_ARGS__)

TEMPLATE_HEADER
void PROPERTY_CLASS::setProperty(int property, const property_impl::PropertyValue &value) {
	switch (property) {
	PROPERTY_LIST
	default:
		PROPERTY_BASE::setProperty(property, value);
	}
}

#undef PROPERTY_CLASS
#undef PROPERTY_BASE
#undef PROPERTY_LIST
#undef INT_P
#undef DOUBLE_P
#undef OBJECT_P
#undef DOUBLE3_P
#undef DOUBLE6_P
#undef HAS_
#undef GET_
#undef _GET_CONV_
#undef SET_
#undef INT_MIN
#undef INT_MAX
#undef FLOAT_MIN
#undef FLOAT_MAX
#undef PROPERTY_CLASS_IS_TEMPLATE
#undef TEMPLATE_HEADER
}
