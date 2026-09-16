// The same expression, x*y + 2*x, in all three designs, over the same data.
//
//   build:  g++ -std=c++23 -O2 -march=native bench.cpp -o bench && ./bench
//
// Each kernel sums the expression over n doubles.  by_value and by_reference
// have to rebuild the tree every iteration -- one holds copies, the other holds
// references to temporaries that die at the semicolon.  in_the_type builds it
// once, outside the loop, because a one-byte object has nothing to dangle.

#include <chrono>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <vector>

namespace by_value {
struct Var { double v; constexpr double eval() const { return v; } };
struct Const { double v; constexpr double eval() const { return v; } };
template <class L, class R> struct Add { L lhs; R rhs;
  constexpr double eval() const { return lhs.eval() + rhs.eval(); } };
template <class L, class R> struct Mul { L lhs; R rhs;
  constexpr double eval() const { return lhs.eval() * rhs.eval(); } };
template <class L, class R> constexpr Add<L, R> operator+(L l, R r) { return {l, r}; }
template <class L, class R> constexpr Mul<L, R> operator*(L l, R r) { return {l, r}; }

double sum(const std::vector<double> &d, double yv) {
  double acc = 0;
  for (double v : d) {
    Var x{v}, y{yv};
    Const two{2.0};
    acc += (x * y + two * x).eval();       // 32 bytes, rebuilt and copied per element
  }
  return acc;
}
}  // namespace by_value

namespace by_reference {
struct Var { double v; double eval() const { return v; } };
struct Const { double v; double eval() const { return v; } };
template <class L, class R> struct Add { const L &lhs; const R &rhs;
  double eval() const { return lhs.eval() + rhs.eval(); } };
template <class L, class R> struct Mul { const L &lhs; const R &rhs;
  double eval() const { return lhs.eval() * rhs.eval(); } };
template <class L, class R> Add<L, R> operator+(const L &l, const R &r) { return {l, r}; }
template <class L, class R> Mul<L, R> operator*(const L &l, const R &r) { return {l, r}; }

double sum(const std::vector<double> &d, double yv) {
  double acc = 0;
  for (double v : d) {
    Var x{v}, y{yv};
    Const two{2.0};
    acc += (x * y + two * x).eval();       // 16 bytes, and only valid inside the statement
  }
  return acc;
}
}  // namespace by_reference

namespace in_the_type {
struct Var { double v; Var &operator=(double d) { v = d; return *this; } };
template <const auto &L> struct Ref { double eval() const { return L.v; } };
template <double V> struct Const { constexpr double eval() const { return V; } };
template <class L, class R> struct Add {
  double eval() const { return L{}.eval() + R{}.eval(); } };
template <class L, class R> struct Mul {
  double eval() const { return L{}.eval() * R{}.eval(); } };
template <class L, class R> constexpr Add<L, R> operator+(L, R) { return {}; }
template <class L, class R> constexpr Mul<L, R> operator*(L, R) { return {}; }

static Var vx{0.0}, vy{0.0};

double sum(const std::vector<double> &d, double yv) {
  constexpr Ref<vx> x;
  constexpr Ref<vy> y;
  constexpr Const<2.0> two;
  constexpr auto e = x * y + two * x;       // 1 byte, built once, outside the loop
  vy = yv;
  double acc = 0;
  for (double v : d) {
    vx = v;
    acc += e.eval();
  }
  return acc;
}
}  // namespace in_the_type

// Every rep gets a different y, so the sum cannot be computed once and reused,
// and the result is forced into a register so it cannot be discarded.
template <class F> double time_ms(F f, int reps) {
  double sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    sink += f(3.0 + r);
    asm volatile("" : : "r"(&sink) : "memory");
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf("%14.3f", std::chrono::duration<double, std::milli>(t1 - t0).count() / reps);
  return sink;
}

// bench            -- the table, all three designs
// bench <design> <n> <reps>   -- one design only, so perf can attribute to it
int main(int argc, char **argv) {
  const int n = argc > 2 ? std::atoi(argv[2]) : 1 << 20;
  const int reps = argc > 3 ? std::atoi(argv[3]) : 200;
  std::vector<double> d(n);
  for (int i = 0; i < n; ++i) d[i] = 1.0 + i % 7;

  if (argc > 1) {
    const std::string which = argv[1];
    double keep = 0;
    if (which == "value") keep = time_ms([&](double yv) { return by_value::sum(d, yv); }, reps);
    else if (which == "ref") keep = time_ms([&](double yv) { return by_reference::sum(d, yv); }, reps);
    else keep = time_ms([&](double yv) { return in_the_type::sum(d, yv); }, reps);
    std::printf("  %s  checksum %.1f\n", which.c_str(), keep);
    return 0;
  }

  std::printf("%-16s %10s %14s\n", "design", "sizeof", "ms/pass");
  double keep = 0;
  std::printf("%-16s %10zu", "by value", sizeof(by_value::Var{} * by_value::Var{}
                                              + by_value::Const{} * by_value::Var{}));
  keep += time_ms([&](double yv) { return by_value::sum(d, yv); }, reps);
  std::printf("\n%-16s %10zu", "by reference", sizeof(by_reference::Var{} * by_reference::Var{}));
  keep += time_ms([&](double yv) { return by_reference::sum(d, yv); }, reps);
  std::printf("\n%-16s %10zu", "in the type", sizeof(in_the_type::Ref<in_the_type::vx>{}
                                                   * in_the_type::Ref<in_the_type::vy>{}));
  keep += time_ms([&](double yv) { return in_the_type::sum(d, yv); }, reps);
  std::printf("\n\nchecksum %.1f\n", keep);
}
