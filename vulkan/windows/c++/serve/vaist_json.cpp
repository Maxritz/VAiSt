/**
 * \file vaist_json.cpp
 * \brief Implementation of vaist_json.hpp — JSON builder and parser.
 */
#include "vaist_json.hpp"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace vaist {

/* ======================================================================== */
/* JsonValue::dump / dump_pretty                                              */
/* ======================================================================== */

std::string JsonValue::escape_string(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<int>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

std::string JsonValue::format_double(double d) {
    if (std::isnan(d) || std::isinf(d)) {
        return "null";
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", d);
    return buf;
}

std::string JsonValue::dump() const {
    switch (type_) {
        case JsonType::kNull:   return "null";
        case JsonType::kBool:   return bool_ ? "true" : "false";
        case JsonType::kInt:    return std::to_string(int_);
        case JsonType::kDouble: return format_double(double_);
        case JsonType::kString: return std::string("\"") + escape_string(str_) + "\"";
        case JsonType::kArray: {
            std::string out = "[";
            for (size_t i = 0; i < arr_.size(); i++) {
                if (i) out += ", ";
                out += arr_[i].dump();
            }
            out += "]";
            return out;
        }
        case JsonType::kObject: {
            std::string out = "{";
            for (size_t i = 0; i < obj_.size(); i++) {
                if (i) out += ", ";
                out += escape_string(obj_[i].first);
                out += ": ";
                out += obj_[i].second.dump();
            }
            out += "}";
            return out;
        }
    }
    return "null";
}

std::string JsonValue::dump_pretty(int indent) const {
    const int spaces = indent * 2;
    std::string pad(spaces, ' ');
    std::string pad2(spaces + 2, ' ');

    switch (type_) {
        case JsonType::kNull:   return "null";
        case JsonType::kBool:   return bool_ ? "true" : "false";
        case JsonType::kInt:    return std::to_string(int_);
        case JsonType::kDouble: return format_double(double_);
        case JsonType::kString: return std::string("\"") + escape_string(str_) + "\"";
        case JsonType::kArray: {
            if (arr_.empty()) return "[]";
            std::string out = "[\n";
            for (size_t i = 0; i < arr_.size(); i++) {
                if (i) out += ",\n";
                out += pad2 + arr_[i].dump_pretty(indent + 1);
            }
            out += "\n" + pad + "]";
            return out;
        }
        case JsonType::kObject: {
            if (obj_.empty()) return "{}";
            std::string out = "{\n";
            for (size_t i = 0; i < obj_.size(); i++) {
                if (i) out += ",\n";
                out += pad2 + escape_string(obj_[i].first) + ": " +
                       obj_[i].second.dump_pretty(indent + 1);
            }
            out += "\n" + pad + "}";
            return out;
        }
    }
    return "null";
}

/* ======================================================================== */
/* JsonWriter                                                               */
/* ======================================================================== */

void JsonWriter::ensure_comma() {
    if (needs_comma_.empty()) return;
    if (needs_comma_.back()) {
        buf_ += ',';
        needs_comma_.back() = false;
    }
}

void JsonWriter::start_object() {
    ensure_comma();
    buf_ += '{';
    is_array_.push_back(false);
    needs_comma_.push_back(false);
}

void JsonWriter::end_object() {
    buf_ += '}';
    if (!is_array_.empty()) {
        is_array_.pop_back();
        needs_comma_.pop_back();
    }
    if (!needs_comma_.empty()) {
        needs_comma_.back() = true;
    }
}

void JsonWriter::start_array() {
    ensure_comma();
    buf_ += '[';
    is_array_.push_back(true);
    needs_comma_.push_back(false);
}

void JsonWriter::end_array() {
    buf_ += ']';
    if (!is_array_.empty()) {
        is_array_.pop_back();
        needs_comma_.pop_back();
    }
    if (!needs_comma_.empty()) {
        needs_comma_.back() = true;
    }
}

void JsonWriter::key(std::string_view k) {
    ensure_comma();
    buf_ += '"';
    buf_ += JsonValue::escape_string(k);
    buf_ += "\":";
}

void JsonWriter::value(bool b) {
    ensure_comma();
    buf_ += b ? "true" : "false";
    if (!needs_comma_.empty()) needs_comma_.back() = true;
}

void JsonWriter::value(int64_t i) {
    ensure_comma();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(i));
    buf_ += buf;
    if (!needs_comma_.empty()) needs_comma_.back() = true;
}

void JsonWriter::value(double d) {
    ensure_comma();
    buf_ += JsonValue::format_double(d);
    if (!needs_comma_.empty()) needs_comma_.back() = true;
}

void JsonWriter::value(std::string_view s) {
    ensure_comma();
    buf_ += '"';
    buf_ += JsonValue::escape_string(s);
    buf_ += '"';
    if (!needs_comma_.empty()) needs_comma_.back() = true;
}

void JsonWriter::value(std::nullptr_t) {
    ensure_comma();
    buf_ += "null";
    if (!needs_comma_.empty()) needs_comma_.back() = true;
}

void JsonWriter::reset() {
    buf_.clear();
    is_array_.clear();
    needs_comma_.clear();
    need_key_ = false;
}

/* ======================================================================== */
/* JsonParser                                                               */
/* ======================================================================== */

JsonValue JsonParser::parse(const char* text) {
    JsonParser p(text, text + std::strlen(text));
    p.skip_ws();
    return p.parse_value();
}

JsonValue JsonParser::parse(std::string_view text) {
    const char* start = text.data();
    const char* end = start + text.size();
    JsonParser p(start, end);
    p.skip_ws();
    return p.parse_value();
}

void JsonParser::skip_ws() {
    while (pos_ < end_ && (*pos_ == ' ' || *pos_ == '\t' || *pos_ == '\n' || *pos_ == '\r')) {
        pos_++;
    }
}

JsonValue JsonParser::parse_value() {
    skip_ws();
    if (pos_ >= end_) throw std::runtime_error("Unexpected end of JSON input");
    char c = *pos_;
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    if (c == 't' || c == 'f' || c == 'n') return parse_literal();
    throw std::runtime_error("Unexpected character in JSON");
}

JsonValue JsonParser::parse_object() {
    pos_++; // skip '{'
    JsonValue obj;
    obj.make_object();
    skip_ws();
    if (pos_ < end_ && *pos_ == '}') {
        pos_++;
        return obj;
    }
    while (pos_ < end_) {
        skip_ws();
        if (pos_ >= end_) break;
        if (*pos_ == '}') { pos_++; break; }
        if (*pos_ != '"') throw std::runtime_error("Expected string key in object");
        std::string key = parse_string().as_string();
        skip_ws();
        if (pos_ >= end_ || *pos_ != ':') throw std::runtime_error("Expected ':' after key");
        pos_++;
        JsonValue val = parse_value();
        obj.add_pair(std::move(key), std::move(val));
        skip_ws();
        if (pos_ < end_ && *pos_ == ',') { pos_++; continue; }
        if (pos_ < end_ && *pos_ == '}') { pos_++; break; }
    }
    return obj;
}

JsonValue JsonParser::parse_array() {
    pos_++; // skip '['
    JsonValue arr;
    arr.make_array();
    skip_ws();
    if (pos_ < end_ && *pos_ == ']') {
        pos_++;
        return arr;
    }
    while (pos_ < end_) {
        skip_ws();
        if (pos_ >= end_) break;
        JsonValue val = parse_value();
        arr.add_value(std::move(val));
        skip_ws();
        if (pos_ < end_ && *pos_ == ',') { pos_++; continue; }
        if (pos_ < end_ && *pos_ == ']') { pos_++; break; }
    }
    return arr;
}

JsonValue JsonParser::parse_string() {
    return JsonValue(parse_escaped_string());
}

std::string JsonParser::parse_escaped_string() {
    if (pos_ >= end_ || *pos_ != '"') throw std::runtime_error("Expected '\"' at start of string");
    pos_++; // skip opening quote
    std::string out;
    while (pos_ < end_ && *pos_ != '"') {
        char c = *pos_++;
        if (c == '\\') {
            if (pos_ >= end_) throw std::runtime_error("Unterminated escape in string");
            char esc = *pos_++;
            switch (esc) {
                case '"':  out += '"'; break;
                case '\\': out += '\\'; break;
                case '/':  out += '/'; break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    if (end_ - pos_ < 4) throw std::runtime_error("Incomplete \\u escape");
                    unsigned int code = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = pos_[i];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= (h - '0');
                        else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
                        else throw std::runtime_error("Invalid hex digit in \\u escape");
                    }
                    pos_ += 4;
                    // Encode as UTF-8
                    if (code < 0x80) {
                        out += static_cast<char>(code);
                    } else if (code < 0x800) {
                        out += static_cast<char>(0xC0 | (code >> 6));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (code >> 12));
                        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default: throw std::runtime_error("Invalid escape character");
            }
        } else {
            out += c;
        }
    }
    if (pos_ >= end_ || *pos_ != '"') throw std::runtime_error("Unterminated string");
    pos_++; // skip closing quote
    return out;
}

JsonValue JsonParser::parse_number() {
    const char* start = pos_;
    bool is_float = false;
    if (*pos_ == '-') pos_++;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') pos_++;
    if (pos_ < end_ && *pos_ == '.') {
        is_float = true;
        pos_++;
        while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') pos_++;
    }
    if (pos_ < end_ && (*pos_ == 'e' || *pos_ == 'E')) {
        is_float = true;
        pos_++;
        if (pos_ < end_ && (*pos_ == '+' || *pos_ == '-')) pos_++;
        while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') pos_++;
    }
    std::string num(start, pos_ - start);
    if (is_float) {
        try {
            return JsonValue(std::stod(num));
        } catch (...) {
            throw std::runtime_error("Invalid floating-point number");
        }
    }
    try {
        return JsonValue(static_cast<int64_t>(std::stoll(num)));
    } catch (...) {
        throw std::runtime_error("Invalid integer number");
    }
}

JsonValue JsonParser::parse_literal() {
    if (end_ - pos_ >= 4 && std::memcmp(pos_, "true", 4) == 0) {
        pos_ += 4;
        return JsonValue(true);
    }
    if (end_ - pos_ >= 5 && std::memcmp(pos_, "false", 5) == 0) {
        pos_ += 5;
        return JsonValue(false);
    }
    if (end_ - pos_ >= 4 && std::memcmp(pos_, "null", 4) == 0) {
        pos_ += 4;
        return JsonValue();
    }
    throw std::runtime_error("Invalid JSON literal");
}

} // namespace vaist
