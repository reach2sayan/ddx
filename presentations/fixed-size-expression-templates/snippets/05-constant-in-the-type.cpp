// C++20 finished the job: a double is a template parameter too, so the constant
// needs no object for a Ref to point at.

struct Var {                               // untouched -- still a variable, still 8 bytes
  double v;
  constexpr Var &operator=(double d) { v = d; return *this; }
};

// A Ref still reads a mutable object, so its eval() still cannot be constexpr.
template <const auto &L> struct Ref {
  double eval() const { return L.v; }
};

template <double V> struct Const {
  constexpr double eval() const { return V; }   // no object to read: this folds
};

template <class L, class R> struct Add {
  constexpr double eval() const { return L{}.eval() + R{}.eval(); }
};

template <class L, class R> struct Mul {
  constexpr double eval() const { return L{}.eval() * R{}.eval(); }
};

template <class L, class R> constexpr Add<L, R> operator+(L, R) { return {}; }
template <class L, class R> constexpr Mul<L, R> operator*(L, R) { return {}; }

int main() {
  static Var vx{2.0}, vy{3.0};             // only the variables need an object now

  constexpr Ref<vx> x;
  constexpr Ref<vy> y;
  constexpr Const<2.0> two;                // nothing to refer to: the value is the type

  static_assert(sizeof(two) == 1);
  static_assert(sizeof(x * y + two * x) == 1);
  static_assert((two * two).eval() == 4.0);  // constants alone fold at compile time

  if ((x * y + two * x).eval() != 10.0) return 1;
  vx = 4.0;
  return int((x * y + two * x).eval());      // 20, from an expression that stores nothing
}
