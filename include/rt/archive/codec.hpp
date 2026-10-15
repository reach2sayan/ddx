#pragma once

#include "rt/opcode.hpp"

#include <boost/describe.hpp>
#include <boost/endian/conversion.hpp>
#include <boost/hash2/flavor.hpp>
#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/integer.hpp>
#include <boost/mp11/algorithm.hpp>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>
#include <tuple>
#include <string>
#include <type_traits>
#include <vector>

// The byte codec: how one value becomes bytes and back, and the single
// traversal that drives both directions.  Nothing here knows what a file is.
namespace ddx::rt::detail {

// --- wire scalars -----------------------------------------------------------
using wire_flavor = boost::hash2::little_endian_flavor;

template <std::size_t N>
using wire_uint_t = typename boost::uint_t<8 * N>::exact;

template <typename U>
concept CWireScalar = std::integral<std::remove_cvref_t<U>> ||
                      std::floating_point<std::remove_cvref_t<U>> ||
                      std::is_enum_v<std::remove_cvref_t<U>>;

// The same bits as an unsigned of the same width, whatever the scalar is.
template <typename U>
[[nodiscard]] constexpr auto to_bits(const U &v) noexcept {
  return std::bit_cast<wire_uint_t<sizeof(U)>>(v);
}

// A forged byte must not become an invalid bool, so that one is a compare.
template <typename U>
[[nodiscard]] constexpr U from_bits(wire_uint_t<sizeof(U)> w) noexcept {
  if constexpr (std::same_as<U, bool>) {
    return w != 0;
  } else {
    return std::bit_cast<U>(w);
  }
}

template <typename V> constexpr bool is_vector_v = false;
template <typename U, typename A>
constexpr bool is_vector_v<std::vector<U, A>> = true;
// A fixed count travels in the schema, not the payload.
template <typename V> constexpr bool is_array_v = false;
template <typename U, std::size_t N>
constexpr bool is_array_v<std::array<U, N>> = true;

// Every member, a base class's included: ExternEntry carries its record as a
// base and its function -- which no file holds -- beside it.
template <typename U>
using described_members = boost::describe::describe_members<
    U, boost::describe::mod_any_access | boost::describe::mod_inherited>;

// Whether a vector moves in one memcpy, which the colouring's colours * n
// tables are what make worth having.  A wire scalar is exactly its unsigned's
// width, so the element needs no further test.
template <typename E>
constexpr bool bulk_v =
    CWireScalar<E> && !std::same_as<E, OpCode> && !std::same_as<E, bool> &&
    std::endian::native == std::endian::little;

// The fewest bytes an element can occupy: a string still costs its length.
template <typename U>
[[nodiscard]] constexpr std::size_t element_floor() noexcept {
  if constexpr (CWireScalar<U>) {
    return sizeof(wire_uint_t<sizeof(U)>);
  } else {
    return sizeof(std::uint64_t);
  }
}

// What a Codec's derived half supplies.
template <typename D>
concept CWirePort =
    requires(D &d, std::size_t n, std::vector<std::uint64_t> v) {
      { d.ok() } -> std::same_as<bool>;
      { d.count(n) } -> std::same_as<bool>;
      { d.fit(v, n, n) } -> std::same_as<bool>;
      { d.bytes(v.data(), n) } -> std::same_as<bool>;
    };

// A length-prefixed field is a count that travels *out* of the container when
// writing and *into* it when reading, which is what lets one body serve both:
// `n = c.size()` is what the writer means, and what the reader overwrites.
template <typename D> class Codec {
public:
  bool text(auto &s) { return framed(s, 1); }
  bool blob(auto &b) { return framed(b, 1); }

  // Bulk only: the guard used to live in the traversal, where a direct caller
  // missed it and memcpy'd host padding.
  template <typename C>
    requires bulk_v<typename std::remove_cvref_t<C>::value_type>
  bool array(C &v) {
    return framed(v, sizeof(typename std::remove_cvref_t<C>::value_type));
  }

  // The element loop's half: agree a count and make room, move no bytes.
  template <typename C> bool each(C &v) {
    return sized(v,
                 element_floor<typename std::remove_cvref_t<C>::value_type>());
  }

protected:
  Codec() = default; // only a derived class builds one

private:
  [[nodiscard]] D &self() noexcept { return static_cast<D &>(*this); }

  template <typename C> bool sized(C &c, std::size_t element) {
    std::size_t n = c.size(); // the reading pass overwrites it
    return self().count(n) && self().fit(c, n, element);
  }
  // std::data() yields `const E*` writing and `E*` reading, which is exactly
  // what each bytes() overload takes: constness is checked by instantiation
  // rather than tested for.
  template <typename C> bool framed(C &c, std::size_t element) {
    return sized(c, element) && self().bytes(std::data(c), c.size() * element);
  }
};

class Writer : public Codec<Writer> {
public:
  explicit Writer(std::vector<std::byte> &into) noexcept : out_(&into) {}
  bool scalar(const CWireScalar auto &v) {
    const auto w = boost::endian::native_to_little(to_bits(v));
    return bytes(&w, sizeof w);
  }

  // A Writer cannot run short;
  [[nodiscard]] static constexpr bool ok() noexcept { return true; }
  bool count(std::size_t &n) { return scalar(static_cast<std::uint64_t>(n)); }

  // Nothing to make room for: the container already holds what is going out.
  template <typename C>
  static constexpr bool fit(C &, std::size_t, std::size_t) noexcept {
    return true;
  }

  bool bytes(const void *from, std::size_t n) {
    const auto *const p = static_cast<const std::byte *>(from);
    out_->insert(out_->end(), p, p + n);
    return true;
  }

private:
  std::vector<std::byte> *out_;
};
static_assert(CWirePort<Writer>);

// Prices a value before a Writer writes it: the same traversal with no bytes
// moved, so an encode can reserve exactly once.
class Counter : public Codec<Counter> {
public:
  bool scalar(const CWireScalar auto &v) {
    total_ += sizeof(to_bits(v));
    return true;
  }

  [[nodiscard]] static constexpr bool ok() noexcept { return true; }
  bool count(std::size_t &) {
    total_ += sizeof(std::uint64_t);
    return true;
  }
  template <typename C>
  static constexpr bool fit(C &, std::size_t, std::size_t) noexcept {
    return true;
  }
  bool bytes(const void *, std::size_t n) {
    total_ += n;
    return true;
  }

  [[nodiscard]] std::size_t total() const noexcept { return total_; }

private:
  std::size_t total_ = 0;
};
static_assert(CWirePort<Counter>);

class Reader : public Codec<Reader> {
public:
  static constexpr std::uint8_t no_op = 0xFF;
  static_assert(op_count < no_op);

  // `remap` turns the file's opcode byte into this build's, or no_op where the
  // file names one this build does not have.
  Reader(std::span<const std::byte> from,
         std::span<const std::uint8_t> remap) noexcept
      : in_{from}, remap_{remap} {}

  bool scalar(CWireScalar auto &v) {
    using S = std::remove_cvref_t<decltype(v)>;
    wire_uint_t<sizeof(S)> w{};
    if (!bytes(&w, sizeof w)) {
      return false;
    }
    boost::endian::little_to_native_inplace(w);
    if constexpr (std::same_as<S, OpCode>) {
      const auto i = static_cast<std::size_t>(w);
      const std::uint8_t mapped = i < remap_.size() ? remap_[i] : no_op;
      if (mapped == no_op) {
        bad_ = true;
        return false;
      }
      v = static_cast<OpCode>(mapped);
    } else {
      v = from_bits<S>(w);
    }
    return true;
  }

  [[nodiscard]] bool ok() const noexcept { return !bad_; }
  [[nodiscard]] bool spent() const noexcept { return in_.empty(); }
  [[nodiscard]] std::size_t remaining() const noexcept { return in_.size(); }

  bool count(std::size_t &n) {
    std::uint64_t w = 0;
    const bool got = scalar(w);
    n = got ? static_cast<std::size_t>(w) : 0;
    return got;
  }

  // The one plausibility gate, checked before every resize: an enormous length
  // allocates wildly and there is no bad_alloc to catch.  Division, so the
  // product `n * element` never wraps into agreeing.  `element` is sizeof(E)
  // where the bytes follow in bulk and the floor where they do not.
  template <typename C> bool fit(C &c, std::size_t n, std::size_t element) {
    if (element != 0 && n > in_.size() / element) {
      bad_ = true;
      return false;
    }
    c.resize(n);
    return true;
  }

  bool bytes(void *into, std::size_t n) {
    if (bad_ || in_.size() < n) {
      bad_ = true;
      return false;
    }
    if (n != 0) { // memcpy from a null data() is undefined even for zero
      std::memcpy(into, in_.data(), n);
    }
    in_ = in_.subspan(n);
    return true;
  }

private:
  std::span<const std::byte> in_;
  std::span<const std::uint8_t> remap_;
  bool bad_ = false;
};
static_assert(CWirePort<Reader>);

// A described aggregate is its fields in order, a vector its length then its
// elements, a string and a blob their own length, everything else a scalar.
// Both sinks answer the same calls, so there is no second implementation to
// drift from; `V` is const on the writing pass only, and direction is a
// property of the sink rather than of the constness.
template <typename Sink, typename V> void wire(Sink &s, V &v) {
  using U = std::remove_const_t<V>;
  if constexpr (std::same_as<U, std::string>) {
    s.text(v);
  } else if constexpr (std::same_as<U, std::vector<std::byte>>) {
    s.blob(v);
  } else if constexpr (CWireScalar<U>) {
    s.scalar(v);
  } else if constexpr (is_vector_v<U>) {
    // Nested, not one `&&`: the bulk test names U::value_type and both
    // arguments would be instantiated.
    if constexpr (bulk_v<typename U::value_type>) {
      s.array(v);
    } else {
      if (!s.each(v)) {
        return;
      }
      for (auto &e : v) {
        wire(s, e);
        if (!s.ok()) {
          return;
        }
      }
    }
  } else if constexpr (is_array_v<U>) {
    for (auto &e : v) {
      wire(s, e);
      if (!s.ok()) {
        return;
      }
    }
  } else {
    static_assert(boost::describe::has_describe_members<U>::value,
                  "rt::archive: no described field list for this type");
    boost::mp11::mp_for_each<described_members<U>>([&](auto D) {
      if (s.ok()) {
        wire(s, v.*D.pointer);
      }
    });
  }
}

// The bytes wire() will write for these values, asked of the Counter: what an
// encode reserves before its Writer runs the same traversal.
template <typename... Vs>
[[nodiscard]] std::size_t wire_size(const Vs &...vs) {
  Counter c;
  (wire(c, std::as_const(vs)), ...);
  return c.total();
}

// Field names and leaf widths folded in traversal order and carried in the
// prologue, so a moved or retyped field refuses old files.
template <typename V> constexpr void fold_schema(boost::hash2::fnv1a_32 &h) {
  using U = std::remove_const_t<V>;
  // A view, never the char array: hash_append would carry the array's NUL.
  const auto text = [&h](std::string_view s) {
    boost::hash2::hash_append(h, wire_flavor{}, s);
  };
  if constexpr (std::same_as<U, std::string>) {
    text("str");
  } else if constexpr (std::same_as<U, std::vector<std::byte>>) {
    text("blob");
  } else if constexpr (CWireScalar<U>) {
    text(std::floating_point<U> ? "f" : "u");
    boost::hash2::hash_append(
        h, wire_flavor{},
        static_cast<std::uint32_t>(sizeof(wire_uint_t<sizeof(U)>)));
  } else if constexpr (is_vector_v<U>) {
    text("vec<");
    fold_schema<typename U::value_type>(h);
    text(">");
  } else if constexpr (is_array_v<U>) {
    text("arr<");
    boost::hash2::hash_append(h, wire_flavor{},
                              static_cast<std::uint32_t>(std::tuple_size_v<U>));
    fold_schema<typename U::value_type>(h);
    text(">");
  } else {
    static_assert(boost::describe::has_describe_members<U>::value);
    text("{");
    boost::mp11::mp_for_each<described_members<U>>([&](auto D) {
      text(D.name);
      text(":");
      fold_schema<
          std::remove_reference_t<decltype(std::declval<U &>().*D.pointer)>>(h);
      text(";");
    });
    text("}");
  }
}

// Several shapes fold into one stamp in order: the prologue's own shape rides
// beside the payload's, so adding a header field refuses old files by itself.
template <typename... Vs> [[nodiscard]] constexpr std::uint32_t schema_of() {
  boost::hash2::fnv1a_32 h;
  (fold_schema<Vs>(h), ...);
  return h.result();
}

// The bytes a described aggregate of scalars occupies.  Only the prologue asks:
// everything else is length-prefixed and has no fixed size.
template <typename V> [[nodiscard]] constexpr std::size_t wire_bytes() {
  using U = std::remove_const_t<V>;
  if constexpr (CWireScalar<U>) {
    return sizeof(wire_uint_t<sizeof(U)>);
  } else {
    static_assert(boost::describe::has_describe_members<U>::value);
    std::size_t n = 0;
    boost::mp11::mp_for_each<described_members<U>>([&](auto D) {
      n += wire_bytes<
          std::remove_reference_t<decltype(std::declval<U &>().*D.pointer)>>();
    });
    return n;
  }
}

} // namespace ddx::rt::detail
