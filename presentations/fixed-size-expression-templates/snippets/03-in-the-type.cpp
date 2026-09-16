// The reference moves into the type: the node stores nothing, so nothing can dangle.

struct Var {                               // unchanged: a variable you can assign to
  double v;
  constexpr Var &operator=(double d) { v = d; return *this; }
};

struct Const {                             // unchanged: it holds its value,
  double v;                                // and nothing can assign to it
  void operator=(double) = delete;
};

// C++17 made this leaf legal: `auto` template parameters, and a referenced object
// that only needs static storage duration -- no linkage, so a local static will do.
template <const auto &L> struct Ref {      // the reference is the type, not a member
  double eval() const { return L.v; }      // not constexpr: it reads a mutable object
};

template <class L, class R> struct Add {                        // no data members
  double eval() const { return L{}.eval() + R{}.eval(); }       // nor here: L and R are Refs
};

template <class L, class R> struct Mul {                        // no data members
  double eval() const { return L{}.eval() * R{}.eval(); }
};

template <class L, class R> constexpr Add<L, R> operator+(L, R) { return {}; }
template <class L, class R> constexpr Mul<L, R> operator*(L, R) { return {}; }

int main() {
  static Var vx{2.0}, vy{3.0};             // the leaves live where you declared them
  static Const c2{2.0};

  constexpr Ref<vx> x;
  constexpr Ref<vy> y;
  constexpr Ref<c2> two;

  static_assert(sizeof(x) == 1);
  static_assert(sizeof(x * y) == 1);
  static_assert(sizeof(x * y + two * x) == 1);
  static_assert(sizeof(x * y + two * x + x * y) == 1);

  auto e = x * y + two * x;                // one byte, and it outlives every temporary
  if (e.eval() != 10.0) return 1;
  vx = 4.0;
  return int(e.eval());                    // 20: the same one-byte object, the new value
}
