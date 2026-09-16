// Children stored by value: safe, but the object grows with the expression.

struct Var {
  double v;
  constexpr Var &operator=(double d) { v = d; return *this; }   // a variable can change
  constexpr double eval() const { return v; }
};

struct Const {
  double v;
  void operator=(double) = delete;                              // a constant cannot
  constexpr double eval() const { return v; }
};

template <class L, class R> struct Add {
  L lhs; R rhs;                                                 // both children stored
  constexpr double eval() const { return lhs.eval() + rhs.eval(); }
};

template <class L, class R> struct Mul {
  L lhs; R rhs;
  constexpr double eval() const { return lhs.eval() * rhs.eval(); }
};

template <class L, class R> constexpr Add<L, R> operator+(L l, R r) { return {l, r}; }
template <class L, class R> constexpr Mul<L, R> operator*(L l, R r) { return {l, r}; }

int main() {
  Var x{2.0}, y{3.0};
  Const two{2.0};

  static_assert(sizeof(x) == 8);
  static_assert(sizeof(x * y) == 16);
  static_assert(sizeof(x * y + two * x) == 32);        // one expression, four nodes
  static_assert(sizeof(x * y + two * x + x * y) == 48);

  auto e = x * y + two * x;
  if (e.eval() != 10.0) return 1;
  x = 4.0;
  return int(e.eval());                  // still 10: the tree copied x, it never watched it
}
