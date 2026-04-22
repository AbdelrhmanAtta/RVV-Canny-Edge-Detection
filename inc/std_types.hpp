#pragma once

#include <cstdint>
#include <expected>

/**
 * @file std_types.hpp
 * @brief Standard type definitions and common status codes.
 *
 * This header provides basic platform-independent types and return
 * status values used across the project.
 */

/**
 * @enum Status
 * @brief Represents the execution result of a function.
 *
 * Used to indicate whether an operation succeeded or failed.
 */
enum class Status : uint8_t
{
    E_OK  = 0,              /**< Operation completed successfully */
    E_INVAL_PTR,            /**< Invalid Pointer */
    E_ALLOC_FAIL,           /**< Memory Allocation Failure */
    E_INVAL_DIR,            /**< Invalid Direction */
};
