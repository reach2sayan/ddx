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

// A C-contiguous NumPy array of shape (symbols, points) already matches the
// kernel ABI's column layout, so inputs and outputs are used without copying.
namespace ddx::py {

namespace pyb = pybind11;

class PyEquation;
class PyCall;
class PyRestricted;

using Array = pyb::array_t<double, pyb::array::c_style | pyb::array::forcecast>;

[[nodiscard]] constexpr pyb::ssize_t ssize(std::size_t n) noexcept {
  return static_cast<pyb::ssize_t>(n);
}

// Rank is at most (function, symbol, symbol) plus a batch axis, so a shape
// fits on the stack.
using Shape = boost::container::static_vector<pyb::ssize_t, 4>;

// The symbol names as text and as interned Python strings, so that dict
// lookups match by pointer before comparing characters.
class Symbols {
public:
  explicit Symbols(std::span<const std::string> text)
      : text_(text.begin(), text.end()),
        keys_(text | std::views::transform([](const std::string &s) {
                PyObject *key = pyb::str(s).release().ptr();
                PyUnicode_InternInPlace(&key);
                return pyb::reinterpret_steal<pyb::object>(key);
              }) |
              impl::to<std::vector>()),
        listed_{keys_.size()} {
    for (const auto [i, key] : keys_ | std::views::enumerate) {
      listed_[static_cast<std::size_t>(i)] = key;
    }
  }
  [[nodiscard]] std::span<const std::string> text() const noexcept {
    return text_;
  }
  [[nodiscard]] std::span<const pyb::object> keys() const noexcept {
    return keys_;
  }
  // A fresh list over the existing keys, so no new strings are created.
  [[nodiscard]] pyb::list listed() const {
    return pyb::reinterpret_steal<pyb::list>(PySequence_List(listed_.ptr()));
  }
  [[nodiscard]] std::size_t size() const noexcept { return text_.size(); }

private:
  std::vector<std::string> text_;
  std::vector<pyb::object> keys_;
  pyb::tuple listed_; // the keys as one tuple, copied into each listed()
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

  // Not copyable, since columns_ points into the inline storage of scalars_ and
  // filled_.
  Point(const Point &) = delete;
  Point &operator=(const Point &) = delete;

private:
  friend class PyEquation;
  friend class PyCall;
  friend class PyRestricted;

  // A direction given by position only, since functions have no names for a
  // dict to key on.
  Point(const pyb::handle &x, std::size_t count) : columns_(count, nullptr) {
    from_array(x, count);
  }

  [[nodiscard]] std::span<const double *const> columns() const {
    return {columns_.data(), columns_.size()};
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return n_; }
  [[nodiscard]] constexpr bool batched() const noexcept { return batched_; }

  // Fast path for a dict of plain numbers at one point, with no NumPy objects.
  // Returns false for any other input, leaving from_dict to report errors.
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
          return false; // too big for a double; from_dict reports it
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

  // The general dict path, for arrays, batches and anything the fast path
  // declined.
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
      fail_with(errc::unknown_symbol); // a key that names no symbol
    }
    if (found != names.size()) {
      fail_with(errc::wrong_arity); // a symbol is missing
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

    // Each scalar broadcast over the batch gets a run in filled_, which is
    // sized before any pointer into it is taken.
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

  // An array whose rows are the symbols, in order.
  void from_array(const pyb::handle &x, std::size_t arity) {
    auto a = pyb::cast<Array>(x);
    if ((a.ndim() != 1 && a.ndim() != 2) ||
        static_cast<std::size_t>(a.shape(0)) != arity) {
      fail_with(errc::wrong_arity);
    }
    n_ = a.ndim() == 2 ? static_cast<std::size_t>(a.shape(1)) : 1;
    batched_ = a.ndim() == 2;

    std::ranges::transform(
        std::views::iota(0uz, arity), columns_.begin(),
        [n = n_, base = a.data()](std::size_t j) { return base + j * n; });
    held_.push_back(std::move(a));
  }

  boost::container::small_vector<const double *, 16> columns_;
  boost::container::small_vector<double, 16> scalars_; // fast-path values
  boost::container::small_vector<Array, 2> held_;      // arrays columns_ reads
  boost::container::small_vector<double, 16> filled_;  // broadcast scalars
  std::size_t n_ = 1;
  bool batched_ = false;
};

// The table of output column pointers the kernel ABI takes, `n` apart. They
// point into heap storage, so moving the table is safe.
class Columns {
public:
  [[nodiscard]] std::span<double *const> rows() {
    return {rows_.data(), rows_.size()};
  }
  [[nodiscard]] double *at(std::size_t column) { return rows_[column]; }

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

// The ABI writes (columns, n) row-major, which is already the memory of the
// caller's shape, so the array is never reshaped.
class Block : public Columns {
public:
  Block(const Shape &shape, std::size_t columns, std::size_t n)
      : Block{Array{shape}, columns, n} {}

  [[nodiscard]] pyb::object array() && { return std::move(array_); }
  // A bound call reuses this array on every call instead of handing it over.
  [[nodiscard]] const Array &bound() const noexcept { return array_; }

private:
  Block(Array a, std::size_t columns, std::size_t n)
      : Columns{a.mutable_data(), columns, n}, array_{std::move(a)} {}
  Array array_;
};

// A dense output block. Its rows never go to the ABI, so column addresses are
// computed rather than stored.
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
