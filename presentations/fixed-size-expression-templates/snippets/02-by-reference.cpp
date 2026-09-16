// Children stored by reference: fixed size, but the temporaries are gone.

struct Var {
  double v;
  Var &operator=(double d) { v = d; return *this; }   // a variable can change
  double eval() const { return v; }
};

struct Const {
  double v;
  void operator=(double) = delete;                    // a constant cannot
  double eval() const { return v; }
};

template <class L, class R> struct Add {
  const L &lhs; const R &rhs;                         // 8 bytes each, whatever L and R are
  double eval() const { return lhs.eval() + rhs.eval(); }
};

template <class L, class R> struct Mul {
  const L &lhs; const R &rhs;
  double eval() const { return lhs.eval() * rhs.eval(); }
};

template <class L, class R> Add<L, R> operator+(const L &l, const R &r) { return {l, r}; }
template <class L, class R> Mul<L, R> operator*(const L &l, const R &r) { return {l, r}; }

int main() {
  Var x{2.0}, y{3.0};
  Const two{2.0};

  static_assert(sizeof(x * y) == 16);
  static_assert(sizeof(x * y + two * x) == 16);   // fixed, however deep

  x = 4.0;                                        // a reference watches its variable:
  if ((x * y + two * x).eval() != 20.0) return 1; // 20 now, not the 10 it was built with

  auto e = x * y + two * x;                       // both Muls die at this semicolon
  return int(e.eval());                           // read through dangling references
}
