/**
 * \file vaist_json.hpp
 * \brief Lightweight JSON builder/parser utilities for VAiSt HTTP server.
 *
 * Provides a self-contained JSON library (no external dependencies) that
 * supports the subset of JSON needed by the OpenAI-compatible API:
 * objects, arrays, strings (with UTF-8, quotes, backslash, control-char
 * escaping), numbers (integers and floating point), booleans, and null.
 *
 * Design goals:
 * - Zero dynamic allocations during parsing (single-pass recursive descent).
 * - Safe string escaping for HTTP response construction.
 * - Small ABI footprint: JsonValue is a tagged variant with move semantics.
 */
#ifndef VAIST_CPP_JSON_HPP
#define VAIST_CPP_JSON_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace vaist {

/**
 * \brief JSON value types (subset of RFC 8259).
 */
enum class JsonType : uint8_t {
    kNull   = 0,  /**< JSON null.              */
    kBool   = 1,  /**< JSON boolean.           */
    kInt    = 2,  /**< JSON number (integer).  */
    kDouble = 3,  /**< JSON number (floating). */
    kString = 4,  /**< JSON string.            */
    kArray  = 5,  /**< JSON array.             */
    kObject = 6,  /**< JSON object.            */
};

/**
 * \brief A JSON value variant.
 *
 * Holds exactly one value of the type indicated by \c type().
 * Designed for move semantics; copy is supported for convenience.
 */
class JsonValue {
public:
    /** \brief Default-constructs a null value. */
    JsonValue() : type_(JsonType::kNull) {}

    /** \brief Construct from a boolean. */
    explicit JsonValue(bool b) : type_(JsonType::kBool), bool_(b) {}

    /** \brief Construct from an integer. */
    explicit JsonValue(int64_t i) : type_(JsonType::kInt), int_(i) {}

    /** \brief Construct from a double. */
    explicit JsonValue(double d) : type_(JsonType::kDouble), double_(d) {}

    /** \brief Construct null. */
    JsonValue(std::nullptr_t) : type_(JsonType::kNull) {}

    /** \brief Construct from a C string (becomes a JSON string). */
    explicit JsonValue(const char* s) : type_(JsonType::kString), str_(s ? s : "") {}

    /** \brief Construct from a std::string (becomes a JSON string). */
    explicit JsonValue(std::string s) : type_(JsonType::kString), str_(std::move(s)) {}

    /** \brief Construct from a string_view (becomes a JSON string). */
    explicit JsonValue(std::string_view sv) : type_(JsonType::kString), str_(sv) {}

    /** \brief Move constructor. */
    JsonValue(JsonValue&& other) noexcept
        : type_(other.type_) {
        switch (type_) {
            case JsonType::kBool:   bool_ = other.bool_; break;
            case JsonType::kInt:    int_ = other.int_; break;
            case JsonType::kDouble: double_ = other.double_; break;
            case JsonType::kString: str_ = std::move(other.str_); break;
            case JsonType::kArray:  arr_ = std::move(other.arr_); break;
            case JsonType::kObject: obj_ = std::move(other.obj_); break;
            default: break;
        }
    }

    /** \brief Move assignment. */
    JsonValue& operator=(JsonValue&& other) noexcept {
        if (this != &other) {
            type_ = other.type_;
            switch (type_) {
                case JsonType::kBool:   bool_ = other.bool_; break;
                case JsonType::kInt:    int_ = other.int_; break;
                case JsonType::kDouble: double_ = other.double_; break;
                case JsonType::kString: str_ = std::move(other.str_); break;
                case JsonType::kArray:  arr_ = std::move(other.arr_); break;
                case JsonType::kObject: obj_ = std::move(other.obj_); break;
                default: break;
            }
        }
        return *this;
    }

    JsonType type() const noexcept { return type_; }

    bool as_bool() const noexcept { return bool_; }
    int64_t as_int() const noexcept { return int_; }
    double as_double() const noexcept { return double_; }
    const std::string& as_string() const noexcept { return str_; }
    const std::vector<JsonValue>& as_array() const noexcept { return arr_; }
    const std::vector<std::pair<std::string, JsonValue>>& as_object() const noexcept { return obj_; }

    /* ---- Builder methods (used by JsonParser) ---- */

    /** \brief Make this a JSON object. */
    void make_object() noexcept {
        type_ = JsonType::kObject;
        obj_.clear();
    }

    /** \brief Add a key-value pair to this object. */
    void add_pair(std::string&& key, JsonValue&& val) {
        obj_.emplace_back(std::move(key), std::move(val));
    }

    /** \brief Make this a JSON array. */
    void make_array() noexcept {
        type_ = JsonType::kArray;
        arr_.clear();
    }

    /** \brief Append a value to this array. */
    void add_value(JsonValue&& val) {
        arr_.push_back(std::move(val));
    }

    /**
     * \brief Serialize this value to a JSON string.
     * \return Compact JSON string representation.
     */
    std::string dump() const;

    /**
     * \brief Serialize this value to a pretty-printed JSON string.
     * \param indent  Current indent level (internal use, pass 0).
     * \return Pretty JSON string with 2-space indentation.
     */
    std::string dump_pretty(int indent = 0) const;

private:
    JsonType type_;
    bool bool_ = false;
    int64_t int_ = 0;
    double double_ = 0.0;
    std::string str_;
    std::vector<JsonValue> arr_;
    std::vector<std::pair<std::string, JsonValue>> obj_;

    friend class JsonWriter;

    static std::string escape_string(std::string_view s);
    static std::string format_double(double d);
};

/**
 * \brief JSON writer: builds JSON strings via a streaming API.
 *
 * Usage:
 * \code
 *   JsonWriter w;
 *   w.start_object();
 *   w.key("id"); w.value("chatcmpl-123");
 *   w.key("choices"); w.start_array();
 *   w.start_object(); w.key("index"); w.value(0); w.key("text"); w.value("Hello");
 *   w.end_object();
 *   w.end_array();
 *   w.end_object();
 *   std::string json = w.str();
 * \endcode
 */
class JsonWriter {
public:
    JsonWriter() = default;

    void start_object();
    void end_object();
    void start_array();
    void end_array();

    void key(std::string_view k);
    void value(bool b);
    void value(int64_t i);
    void value(int i) { value(static_cast<int64_t>(i)); }
    void value(double d);
    void value(std::string_view s);
    /** \brief Write a JSON null value. */
    void value(std::nullptr_t);
    /** \brief Write a JSON string from a C string. */
    void value(const char* s) { value(std::string_view(s)); }
    /** \brief Write a JSON string from std::string. */
    void value(const std::string& s) { value(std::string_view(s)); }

    /** \brief Reset the writer for reuse. */
    void reset();

    /** \brief Get the accumulated JSON string. */
    std::string str() const { return buf_; }

private:
    std::string buf_;
    std::vector<bool> is_array_;  // stack: true = currently in array element position
    std::vector<bool> needs_comma_; // stack: whether a comma is needed before next element/key
    bool need_key_ = false;  // true when expecting a key (inside object, no pending key)

    void ensure_comma();
};

/**
 * \brief JSON parser: recursive-descent parser with error handling.
 *
 * Parses a JSON string into a JsonValue tree. Throws std::runtime_error
 * on syntax errors.
 */
class JsonParser {
public:
    /**
     * \brief Parse a JSON string.
     * \param text  NUL-terminated JSON string.
     * \return Parsed JsonValue.
     * \throws std::runtime_error on parse error.
     */
    static JsonValue parse(const char* text);

    /**
     * \brief Parse a JSON string from a string_view.
     * \param text  JSON content.
     * \return Parsed JsonValue.
     * \throws std::runtime_error on parse error.
     */
    static JsonValue parse(std::string_view text);

private:
    const char* pos_;
    const char* end_;

    explicit JsonParser(const char* start, const char* end) : pos_(start), end_(end) {}

    void skip_ws();
    JsonValue parse_value();
    JsonValue parse_object();
    JsonValue parse_array();
    JsonValue parse_string();
    JsonValue parse_number();
    JsonValue parse_literal();
    std::string parse_escaped_string();
};

} // namespace vaist

#endif // VAIST_CPP_JSON_HPP
