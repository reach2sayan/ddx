#include "tests_tensor_fixtures.hpp"

TEST(MdLayout, SimplexPackedIsABijectionOnSortedMultiIndices) {
  using T = ddx::impl::md_tensor<double, ddx::impl::uniform_extents_t<4, 3>,
                                 ddx::impl::layout_simplex_packed>;
  EXPECT_EQ(T::size(), 20u);

  constexpr T::mapping_type m{};
  std::set<std::size_t> slots;
  for (const std::size_t i : std::views::iota(0uz, 4uz)) {
    for (const std::size_t j : std::views::iota(i, 4uz)) {
      for (const std::size_t k : std::views::iota(j, 4uz)) {
        const auto s = static_cast<std::size_t>(m(i, j, k));
        EXPECT_LT(s, T::size());
        // Distinct multisets must not collide.
        EXPECT_TRUE(slots.insert(s).second);
        // Every permutation of a multiset must land in the same slot.
        EXPECT_EQ(static_cast<std::size_t>(m(k, j, i)), s);
        EXPECT_EQ(static_cast<std::size_t>(m(j, i, k)), s);
        EXPECT_EQ(static_cast<std::size_t>(m(k, i, j)), s);
      }
    }
  }
  EXPECT_EQ(slots.size(), T::size());
}
TEST(ConstexprContract, MdspanLayerIsConstantEvaluated) {
  constexpr auto simplex = [] {
    ddx::impl::nd_tensor_t<double, 3, 3> t{};
    t[2, 1, 0] = 6.0;
    return (t[0, 1, 2]) + (t[1, 0, 2]);
  }();
  static_assert(simplex == 12.0);

  constexpr auto stacked = [] {
    ddx::impl::nd_stack_t<double, 2, 3, 2> s{};
    s[1, 0, 2] = 3.0;
    return (s[1, 2, 0]) + (s[0, 0, 2]);
  }();
  static_assert(stacked == 3.0);

  SUCCEED(); // the static_asserts above are the test
}
