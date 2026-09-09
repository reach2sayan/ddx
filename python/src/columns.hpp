#pragma once

#include "error.hpp"

#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// NumPy and the kernel ABI already agree: a C-contiguous (symbols, points)
namespace ddx::py {

namespace pyb = pybind11;

class PyEquation;
class PyCall;

using Array = pyb::array_t<double, pyb::array::c_style | pyb::array::forcecast>;

[[nodiscard]] constexpr pyb::ssize_t ssize(std::size_t n) noexcept {
  return static_cast<pyb::ssize_t>(n);
}

// Rank is at most (function, symbol, symbol) plus a batch axis, so a shape
// never leaves the stack.
using Shape = boost::container::static_vector<pyb::ssize_t, 4>;

// The symbol list as text and as the interned Python strings a dict lookup
// keys on: one object, built once, so the two cannot come apart.  Building a
// key per symbol per call was most of what a dict point cost.  Interned, so
// CPython's dict probe hits pointer identity before it compares characters.
class Symbols {
public:
  explicit Symbols(std::span<const std::string> text)
      : text_(text.begin(), text.end()),
        keys_(text | std::views::transform([](const std::string &s) {
                PyObject *key = pyb::str(s).release().ptr();
                PyUnicode_InternInPlace(&key);
                return pyb::reinterpret_steal<pyb::object>(key);
              }) |
              impl::to<std::vector<pyb::object>>()),
        listed_{keys_.size()} {
    for (const auto [i, key] : std::views::enumerate(keys_)) {
      listed_[static_cast<std::size_t>(i)] = key;
    }
  }
  [[nodiscard]] std::span<const std::string> text() const noexcept {
    return text_;
  }
  [[nodiscard]] std::span<const pyb::object> keys() const noexcept {
    return keys_;
  }
  // A fresh list over the keys already built, so an access costs increfs and
  // never a PyUnicode.
  [[nodiscard]] pyb::list listed() const {
    return pyb::reinterpret_steal<pyb::list>(PySequence_List(listed_.ptr()));
  }
  [[nodiscard]] std::size_t size() const noexcept { return text_.size(); }

private:
  std::vector<std::string> text_;
  std::vector<pyb::object> keys_;
  pyb::tuple listed_; // the keys again, as one sequence to copy per access
};

class Point {
public:
  Point(const pyb::handle &x, const Symbols &symbols)
      : columns_(symbols.size(), nullptr) {
    if (pyb::isinstance<pyb::dict>(x)) {
      const auto d = pyb::reinterpret_borrow<pyb::dict>(x);
      if (!scalars_from_dict(d, symbols.keys())) {
        from_dict(d, symbols);
      }
    } else {
      from_array(x, symbols.size());
    }
  }

  // columns_ points into scalars_ and filled_, whose inline storage moves with
  // the object; a Point stays where it was built.
  Point(const Point &) = delete;
  Point &operator=(const Point &) = delete;

private:
  friend class PyEquation;
  friend class PyCall;

  // A direction, by position only: a covector is indexed by function, and
  // functions have no names for a dict to key on.
  Point(const pyb::handle &x, std::size_t count) : columns_(count, nullptr) {
    from_array(x, count);
  }

  [[nodiscard]] constexpr std::span<const double *const> columns() const {
    return {columns_.data(), columns_.size()};
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return n_; }
  [[nodiscard]] constexpr bool batched() const noexcept { return batched_; }

  // The common case: a dict of plain numbers at one point.  One lookup per
  // symbol against an interned key, the value read as a double, and no numpy
  // object anywhere -- where the general path below builds an Array per
  // symbol.  np.float64 subclasses float, so it lands here too.
  //
  // False means "not this shape", not "bad input": anything else falls through
  // to from_dict, which is what reports the errors.  A size mismatch is
  // settled there too, so the two paths cannot disagree.
  [[nodiscard]] bool scalars_from_dict(const pyb::dict &d,
                                       std::span<const pyb::object> names) {
    if (d.size() != names.size()) {
      return false;
    }
    scalars_.resize(names.size());
    for (const auto [i, name] : std::views::enumerate(names)) {
      PyObject *const v = PyDict_GetItemWithError(d.ptr(), name.ptr());
      if (v == nullptr) {
        return PyErr_Occurred() != nullptr ? throw pyb::error_already_set()
                                           : false;
      }
      double &slot = scalars_[static_cast<std::size_t>(i)];
      if (PyFloat_Check(v)) {
        slot = PyFloat_AS_DOUBLE(v);
      } else if (PyLong_Check(v)) {
        slot = PyLong_AsDouble(v);
        if (slot == -1.0 && PyErr_Occurred()) {
          PyErr_Clear();
          return false; // an int too big for a double: report from from_dict
        }
      } else {
        return false;
      }
    }
    std::ranges::transform(scalars_, columns_.begin(),
                           [](const double &v) { return &v; });
    n_ = 1;
    batched_ = false;
    return true;
  }

  // The general dict: arrays, batches, and everything the fast path declined.
  // No prescan -- every symbol found plus a size match is already "every key
  // is a symbol", the same argument the fast path makes.
  void from_dict(const pyb::dict &d, const Symbols &symbols) {
    const auto names = symbols.keys();
    held_.reserve(names.size());
    std::size_t found = 0;
    for (const pyb::object &name : names) {
      PyObject *const v = PyDict_GetItemWithError(d.ptr(), name.ptr());
      if (v == nullptr) {
        if (PyErr_Occurred() != nullptr) {
          throw pyb::error_already_set();
        }
        continue;
      }
      ++found;
      auto a = pyb::cast<Array>(pyb::handle{v});
      if (a.ndim() > 1) {
        fail_with(errc::wrong_arity);
      }
      held_.push_back(std::move(a));
    }
    if (found != d.size()) {
      fail_with(errc::unknown_symbol); // a key no symbol answers to
    }
    if (found != names.size()) {
      fail_with(errc::wrong_arity); // a name is missing, which is not a typo
    }

    std::size_t spread = 0;
    for (const Array &a : held_) {
      if (a.ndim() == 1) {
        const auto len = static_cast<std::size_t>(a.shape(0));
        if (batched_ && len != n_) {
          fail_with(errc::wrong_arity);
        }
        n_ = len;
        batched_ = true;
      } else {
        ++spread;
      }
    }

    // One flat run per broadcast scalar, sized before any pointer is taken.
    filled_.resize(batched_ ? spread * n_ : 0);
    double *fill = filled_.data();
    for (auto [column, a] : std::views::zip(columns_, held_)) {
      if (a.ndim() == 1 || n_ == 1) {
        column = a.data();
      } else {
        column = fill;
        fill = std::ranges::fill_n(fill, static_cast<std::ptrdiff_t>(n_),
                                   *a.data());
      }
    }
  }

  // Rows are in symbol order
  void from_array(const pyb::handle &x, std::size_t arity) {
    auto a = pyb::cast<Array>(x);
    if (a.ndim() != 1 && a.ndim() != 2) {
      fail_with(errc::wrong_arity);
    }
    if (static_cast<std::size_t>(a.shape(0)) != arity) {
      fail_with(errc::wrong_arity);
    }
    n_ = a.ndim() == 2 ? static_cast<std::size_t>(a.shape(1)) : 1;
    batched_ = a.ndim() == 2;

    std::ranges::transform(
        std::views::iota(0uz, arity), columns_.begin(),
        [n=n_, base = a.data()](std::size_t j) { return base + j * n; });
    held_.push_back(std::move(a));
  }

  boost::container::small_vector<const double *, 16> columns_;
  boost::container::small_vector<double, 16> scalars_; // a dict of floats
  boost::container::small_vector<Array, 2> held_;      // what columns_ reads
  boost::container::small_vector<double, 16> filled_; // scalars over a batch
  std::size_t n_ = 1;
  bool batched_ = false;
};

// The pointer table the kernel ABI takes: one per output column, `n` apart.
// The pointers go into the array's heap storage, so the table itself moving is
// harmless.
class Columns {
public:
  [[nodiscard]] constexpr std::span<double *const> rows() {
    return {rows_.data(), rows_.size()};
  }
  [[nodiscard]] constexpr double *at(std::size_t column) {
    return rows_[column];
  }

protected:
  Columns(double *base, std::size_t columns, std::size_t n) {
    rows_.reserve(columns);
    impl::append(rows_, std::views::iota(0uz, columns) |
                            std::views::transform([base, n](std::size_t j) {
                              return base + j * n;
                            }));
  }

private:
  boost::container::small_vector<double *, 32> rows_;
};

// The ABI writes (columns, n) row-major, which is the same memory as the shape
// the caller wants -- so the array is built at that shape once and never
// reshaped.
class Block : public Columns {
public:
  Block(const Shape &shape, std::size_t columns, std::size_t n)
      : Block{Array{shape}, columns, n} {}

  [[nodiscard]] pyb::object array() && { return std::move(array_); }
  // A bound call reads the same array on every call, where the allocating ones
  // hand theirs over once and are done with it.
  [[nodiscard]] const Array &bound() const noexcept { return array_; }

private:
  Block(Array a, std::size_t columns, std::size_t n)
      : Columns{a.mutable_data(), columns, n}, array_{std::move(a)} {}
  Array array_;
};

// A dense output block.  Nothing hands its rows to the ABI, so the column
// table is arithmetic rather than storage.
class Dense {
public:
  Dense(const Shape &shape, std::size_t n) : array_{Array{shape}}, n_{n} {
    base_ = array_.mutable_data();
  }

  [[nodiscard]] constexpr double *at(std::size_t column) {
    return base_ + column * n_;
  }
  [[nodiscard]] pyb::object array() && { return std::move(array_); }
  [[nodiscard]] const Array &bound() const noexcept { return array_; }

private:
  Array array_;
  double *base_;
  std::size_t n_;
};

class Scratch : public Columns {
public:
  Scratch(std::size_t columns, std::size_t n)
      : Scratch{std::vector<double>(columns * n), columns, n} {}

private:
  Scratch(std::vector<double> d, std::size_t columns, std::size_t n)
      : Columns{d.data(), columns, n}, data_{std::move(d)} {}
  std::vector<double> data_;
};

} // namespace ddx::py
