#include "rt/archive/archive.hpp"
#include "rt/equation.hpp"
#include "rt/expressions.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <ios>
#include <latch>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// Saving and loading a built equation: the file must reproduce every answer to
// the bit.  Compared within one binary, node ids being assigned in construction
// order and so numbered differently by another compiler.
namespace {

// A file that removes itself, so a failing EXPECT leaves nothing behind.  The
// name carries a per-process token: a fixed path in the system temp directory
// is shared mutable state between processes, however private it looks.
class Scratch {
public:
  explicit Scratch(std::string name)
      : path_(std::filesystem::temp_directory_path() /
              ("ddx_archive_" + std::move(name) + "_" + token() + ".ddx")) {
    wipe();
  }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  ~Scratch() { wipe(); }

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }
  operator const std::filesystem::path &() const { return path_; } // NOLINT

  [[nodiscard]] std::vector<std::byte> bytes() const {
    // Checked, not dereferenced: `*` on a failed result is undefined, so a
    // vanished file would surface as a segfault rather than as a failure.
    auto read = ddx::rt::detail::Container::read(path_);
    EXPECT_TRUE(read.has_value()) << "scratch file went missing";
    return read ? std::move(*read) : std::vector<std::byte>{};
  }
  void write(std::span<const std::byte> b) const {
    ASSERT_TRUE(ddx::rt::detail::Container::write(path_, b).has_value());
  }

private:
  [[nodiscard]] static const std::string &token() {
    static const std::string t = std::to_string(std::random_device{}());
    return t;
  }
  void wipe() const {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  std::filesystem::path path_;
};

// Coupled, so the Hessian colouring is real work -- the expensive thing the
// file exists to keep.
auto coupled() {
  return ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return exp(x * y) + log(y * z) * sin(x + z) + z * z * x;
  });
}

// Two independent builds over two arenas, not one expression used twice.
constexpr std::array kPoint{1.3, 0.7, 2.1};

void same_answers(const auto &want, const auto &got) {
  ASSERT_EQ(want.arity(), got.arity());
  ASSERT_EQ(want.symbols()->size(), got.symbols()->size());
  EXPECT_TRUE(std::ranges::equal(*want.symbols(), *got.symbols()));

  // Bit-identical: a loaded graph is the same graph, in the same order.
  EXPECT_EQ(*want.evaluate(kPoint), *got.evaluate(kPoint));
  EXPECT_EQ(*want.jacobian(kPoint), *got.jacobian(kPoint));
  EXPECT_EQ(*want.hessian(kPoint), *got.hessian(kPoint));
}

} // namespace

TEST(RtArchive, RoundTripsEveryDerivative) {
  const Scratch file{"roundtrip"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  same_answers(built, *loaded);
  EXPECT_TRUE(loaded->loaded());
  EXPECT_FALSE(built.loaded());
}

// The seeded products are serialised, not re-swept, so a loaded equation must
// answer them without ever running build_hvp_impl.
TEST(RtArchive, RoundTripsTheSeededProducts) {
  const Scratch file{"seeded"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  ASSERT_TRUE(loaded->loaded());

  const std::vector<double> v(*built.arity(), 0.5);
  const std::vector<double> w{2.0};
  const auto point = std::vector<double>(*built.arity(), 0.6);

  EXPECT_EQ(*built.hvp(std::span<const double>{v}, point),
            *loaded->hvp(std::span<const double>{v}, point));
  EXPECT_EQ(*built.vjp(std::span<const double>{w}, point),
            *loaded->vjp(std::span<const double>{w}, point));
  EXPECT_EQ(built.hvp_columns(), loaded->hvp_columns());
  EXPECT_EQ(built.vjp_columns(), loaded->vjp_columns());
}

TEST(RtArchive, RoundTripsASystem) {
  const Scratch file{"system"};
  const auto built = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return std::array{x * y + sin(x), exp(x) - y * y};
  });
  ASSERT_TRUE(built.save(file).has_value());

  const auto loaded = ddx::rt::load<double, 2>(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  EXPECT_EQ(*built.evaluate(1.3, 0.7), *loaded->evaluate(1.3, 0.7));
  EXPECT_EQ(*built.jacobian(1.3, 0.7), *loaded->jacobian(1.3, 0.7));
}

// The output count is in the type, so a file holding some other number is
// refused: reading a two-function file as one mis-sizes every buffer.
TEST(RtArchive, RefusesTheWrongOutputCount) {
  const Scratch file{"outputs"};
  const auto two = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return std::array{x * x, sin(x)};
  });
  ASSERT_TRUE(two.save(file).has_value());

  const auto as_one = ddx::rt::load<double, 1>(file.path());
  ASSERT_FALSE(as_one.has_value());
  EXPECT_EQ(as_one.error().code, ddx::errc::archive_mismatch);
}

TEST(RtArchive, VerifyAnswersBothQuestions) {
  const Scratch file{"verify"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  // Readable by this build...
  EXPECT_TRUE(ddx::rt::verify<>(file.path()).has_value());
  // ...and it is this equation.
  EXPECT_TRUE(built.verify(file.path()).has_value());

  // A different model over the same symbols: it loads, and is not this one.
  const auto other = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return x + y + z;
  });
  EXPECT_TRUE(ddx::rt::verify<>(file.path()).has_value());
  const auto mismatch = other.verify(file.path());
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_EQ(mismatch.error().code, ddx::errc::archive_mismatch);
}

TEST(RtArchive, MissingFileIsNotCorruption) {
  const Scratch file{"missing"}; // constructed, deliberately never written
  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value());
  // The first run of a cache, not a damaged file.
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_io);
}

// Nothing below may abort: under -fno-exceptions a Boost throw would, which is
// why the checksum and the invariants clear before a byte is trusted.
TEST(RtArchive, RefusesATruncatedFile) {
  const Scratch file{"truncated"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();

  // Just past the prologue, so the header parses and the payload does not.
  auto cut = whole;
  cut.resize(ddx::rt::detail::Container::header_bytes + 8);
  file.write(cut);
  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_TRUE(loaded.error().code == ddx::errc::archive_corrupt ||
              loaded.error().code == ddx::errc::bad_archive);
}

TEST(RtArchive, RefusesAFlippedByte) {
  const Scratch file{"flipped"};
  ASSERT_TRUE(coupled().save(file).has_value());
  auto bytes = file.bytes();

  // One bit, in the middle of the payload, where it will land inside a node.
  const std::size_t at =
      ddx::rt::detail::Container::header_bytes +
      (bytes.size() - ddx::rt::detail::Container::header_bytes) / 2;
  bytes[at] ^= std::byte{0x01};
  file.write(bytes);

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt);
}

TEST(RtArchive, RefusesForeignAndFutureFiles) {
  const Scratch file{"foreign"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();

  auto wrong_magic = whole;
  wrong_magic[0] = std::byte{'x'};
  file.write(wrong_magic);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);

  // Bumped in place: the checksum covers the payload, not the prologue, so this
  // is what a future writer's file looks like to us.
  auto future = whole;
  future[ddx::rt::detail::Container::magic_bytes] =
      std::byte{0xFF}; // the first `format` byte
  file.write(future);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);

  // Format 3 numbered symbols alphabetically, so its positional points would
  // mean something else here: refused, not reinterpreted.
  auto older = whole;
  older[ddx::rt::detail::Container::magic_bytes] = std::byte{3};
  file.write(older);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);

  // Letting a reserved byte through would make claiming it later a silent
  // format change.  Named, not a literal: the described field list decides
  // where the tail starts, and a field added to it moves this.
  auto reserved = whole;
  reserved[ddx::rt::detail::Container::header_bytes -
           ddx::rt::detail::Container::reserved_bytes] = std::byte{0x01};
  file.write(reserved);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);
}

TEST(RtArchive, RefusesTheWrongScalar) {
  const Scratch file{"scalar"};
  ASSERT_TRUE(coupled().save(file).has_value());
  // Same file, read as float: the scalar tag and the schema stamp both say no.
  EXPECT_EQ(ddx::rt::verify<float>(file.path()).error().code,
            ddx::errc::bad_archive);
}

// Concurrent writers to one path all succeed, and the file is one of theirs.
//
// What it catches, measured: with a shared staging name the *losing rename
// fails* -- a bare ".tmp" gives ~900 failed writes out of 1280 here.  It is NOT
// a torn-file test.  Tearing was unreachable at 1 KiB, 64 KiB and 1 MiB over
// forty rounds each with the bug present, rename being atomic; the whole-file
// checks state the property but the failing-write assertion does the work.
TEST(RtArchive, ConcurrentWritersToOnePathAllSucceed) {
  const Scratch file{"racing"};
  constexpr std::size_t kWriters = 8;
  static_assert(kWriters >= 2, "one writer races nobody");

  // Built up front: build_hessian_impl appends to its arena, so two threads may
  // not construct over one Builder.  Distinct models, so a file is traceable.
  std::vector<std::vector<std::byte>> payloads;
  for (std::size_t k = 0; k < kWriters; ++k) {
    const auto eq = ddx::rt::equation([k] {
      const auto x = ddx::rt::var("x");
      return x * static_cast<double>(k + 1);
    });
    const Scratch one{"racing_src_" + std::to_string(k)};
    EXPECT_TRUE(eq.save(one).has_value());
    payloads.push_back(one.bytes());
  }

  // Asserted, not assumed: a mixture of two identical payloads is identical to
  // both, and `x * 1.0` folds to `x`, so a tidy-up of these models could make
  // two agree and leave every assertion below green forever.
  //
  // The overlap is *forced* by the latch rather than asserted -- asserting it
  // would be a fact about the scheduler.
  for (std::size_t a = 0; a < payloads.size(); ++a) {
    for (std::size_t b = a + 1; b < payloads.size(); ++b) {
      ASSERT_NE(payloads[a], payloads[b]) << "writers " << a << " and " << b;
    }
  }

  std::latch start{static_cast<std::ptrdiff_t>(kWriters)};
  std::vector<std::jthread> writers;
  writers.reserve(kWriters);
  for (const auto &bytes : payloads) {
    writers.emplace_back([&file, &bytes, &start] {
      start.arrive_and_wait(); // nobody writes until everybody is ready
      for (int spin = 0; spin < 8; ++spin) {
        EXPECT_TRUE(
            ddx::rt::detail::Container::write(file.path(), bytes).has_value());
      }
    });
  }
  writers.clear(); // join

  // Stated but not reliably exercised -- see the note above.
  const auto landed = file.bytes();
  EXPECT_TRUE(std::ranges::any_of(payloads, [&landed](const auto &p) {
    return p == landed;
  })) << "the file is not any single writer's";
  const auto loaded = ddx::rt::load(file.path());
  EXPECT_TRUE(loaded.has_value())
      << "a torn file would fail the checksum: " << loaded.error().code;

  // And nothing was left staged beside it.
  std::error_code ec;
  std::size_t leftovers = 0;
  for (const auto &entry :
       std::filesystem::directory_iterator{file.path().parent_path(), ec}) {
    const auto name = entry.path().filename().string();
    if (name.starts_with(file.path().filename().string()) &&
        name.ends_with(".tmp")) {
      ++leftovers;
    }
  }
  EXPECT_EQ(leftovers, 0u) << "staging files left behind";
}

// --- the cache ---------------------------------------------------------------

TEST(RtArchive, CacheBuildsThenLoads) {
  const Scratch file{"cache"};
  const auto model = [] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x * y) + sin(x) * log(y);
  };

  const auto first = ddx::rt::equation(file.path(), model);
  EXPECT_FALSE(first.loaded()) << "nothing to load on the first run";
  ASSERT_TRUE(std::filesystem::exists(file.path())) << "the first run saves";

  const auto second = ddx::rt::equation(file.path(), model);
  EXPECT_TRUE(second.loaded()) << "the second run takes the sweeps off disk";
  EXPECT_EQ(*first.evaluate(1.3, 0.7), *second.evaluate(1.3, 0.7));
  EXPECT_EQ(*first.jacobian(1.3, 0.7), *second.jacobian(1.3, 0.7));
  EXPECT_EQ(*first.hessian(1.3, 0.7), *second.hessian(1.3, 0.7));
}

TEST(RtArchive, CacheRebuildsWhenTheModelChanges) {
  const Scratch file{"stale"};
  const auto before = ddx::rt::equation(file.path(), [] {
    const auto x = ddx::rt::var("x");
    return x * x;
  });
  EXPECT_FALSE(before.loaded());

  // The key is the arena, so the same symbols over a different model must
  // rebuild and overwrite.
  const auto after = ddx::rt::equation(file.path(), [] {
    const auto x = ddx::rt::var("x");
    return x * x * x;
  });
  EXPECT_FALSE(after.loaded()) << "an edited model is not a cache hit";
  EXPECT_NEAR(*after.evaluate(2.0), 8.0, 1e-15);

  // And what it wrote is now the current model.
  const auto again = ddx::rt::equation(file.path(), [] {
    const auto x = ddx::rt::var("x");
    return x * x * x;
  });
  EXPECT_TRUE(again.loaded()) << "the rebuild overwrote the stale file";
  EXPECT_NEAR(*again.evaluate(2.0), 8.0, 1e-15);
}

TEST(RtArchive, CacheSurvivesACorruptFile) {
  const Scratch file{"cache_corrupt"};
  const auto model = [] {
    const auto x = ddx::rt::var("x");
    return sin(x) * x;
  };
  ASSERT_FALSE(ddx::rt::equation(file.path(), model).loaded());

  auto bytes = file.bytes();
  bytes[bytes.size() - 3] ^= std::byte{0xFF};
  file.write(bytes);

  // A damaged cache is a rebuild, never a refusal and never an abort.
  const auto rebuilt = ddx::rt::equation(file.path(), model);
  EXPECT_FALSE(rebuilt.loaded());
  EXPECT_NEAR(*rebuilt.evaluate(1.3), std::sin(1.3) * 1.3, 1e-15);
  EXPECT_TRUE(ddx::rt::equation(file.path(), model).loaded())
      << "and it left a good file behind";
}

// Every byte of a real file, corrupted every way a single byte can be.  The
// claim is not that these are refused with the right code, only that every one
// *returns*: a parser reading a length out of the payload walks off the end
// when the length is wrong, and there is no throw here to catch it.
// Deterministic, so a failure is a failure every run.
TEST(RtArchive, SurvivesEverySingleByteCorruption) {
  const Scratch file{"fuzz"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();
  ASSERT_GT(whole.size(), ddx::rt::detail::Container::header_bytes);

  // Every offset would be a few hundred thousand loads; a stride samples the
  // prologue densely and the payload throughout.
  std::size_t refused = 0;
  std::size_t accepted = 0;
  for (std::size_t at = 0; at < whole.size();
       at += (at < ddx::rt::detail::Container::header_bytes ? 1 : 97)) {
    for (const std::byte mask :
         {std::byte{0x01}, std::byte{0x80}, std::byte{0xFF}}) {
      auto bytes = whole;
      bytes[at] ^= mask;
      file.write(bytes);
      // The only requirement: it comes back.  Under ASan, also where a read
      // past the payload would be caught.
      if (const auto loaded = ddx::rt::load(file.path())) {
        ++accepted;
      } else {
        ++refused;
      }
    }
  }
  // This proves every single-byte corruption *returns*.  It does not prove the
  // file's meaning is protected: 390 of 4290 get past the checksum, and none
  // was the one bit that remapped an opcode -- see RefusesAFlippedOpcodeLabel.
  EXPECT_GT(refused, 0u);
  EXPECT_EQ(
      refused + accepted,
      3 * (ddx::rt::detail::Container::header_bytes +
           (whole.size() - ddx::rt::detail::Container::header_bytes + 96) /
               97));
}

// Every prefix of a real file, which is what a partial write leaves behind.
TEST(RtArchive, SurvivesEveryTruncation) {
  const Scratch file{"truncations"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();

  for (std::size_t keep = 0; keep < whole.size(); keep += 13) {
    auto cut = whole;
    cut.resize(keep);
    file.write(cut);
    // No prefix of a valid file is a valid file: the prologue's payload length
    // no longer matches what follows it.
    EXPECT_FALSE(ddx::rt::load(file.path()).has_value()) << "kept " << keep;
  }
}

// A payload whose checksum agrees and whose contents are still a lie -- the
// case neither fuzz loop can reach, since damage is caught by the checksum long
// before anything parses it.  Written through the real serialiser, which
// recomputes the CRC.
//
// `count` indexes the two colours-by-n tables, and `count * nsym` is where a
// forged one wraps into agreeing with a length.  Only an even symbol count can
// do it: an odd nsym makes the multiply invertible modulo 2^64.
TEST(RtArchive, RefusesAForgedColouringCount) {
  const Scratch file{"forged"};
  const auto four = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto c = ddx::rt::var("c");
    const auto d = ddx::rt::var("d");
    return exp(a * b) + log(c * d) * sin(a + c) + b * d;
  });
  ASSERT_TRUE(four.save(file).has_value());

  auto sound = ddx::rt::load_snapshot<>(file.path());
  ASSERT_TRUE(sound.has_value());
  auto snap = std::move(*sound).release();
  auto &coloring = snap.hessians.front().coloring;
  const std::size_t nsym = snap.symbols.size();

  // There is no `count` field to forge: the colour count is scatter.size() /
  // nsym, so a count that disagrees with its own table cannot be written down.
  // What is left is the grid, and a table that is not one has no colour count
  // at all -- by_color would read past the end of it.
  ASSERT_NE(coloring.scatter.size() % nsym, 1u) << "the forgery must change it";
  coloring.scatter.push_back(ddx::rt::no_column);

  // Through the real serialiser, so the checksum agrees with the lie.
  ASSERT_TRUE(ddx::rt::save(snap, file.path()).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value()) << "a scatter table that is not a grid";
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt);
}

// hash2 appends a float as bit_cast<uint64_t>(v + 0), which folds -0.0 onto
// +0.0.  The digest is the object cache's filename, so agreeing here would hand
// a graph holding -0.0 the kernel compiled for +0.0 -- 1/x apart, and silently.
// digest() routes every field through to_bits for exactly this reason.
TEST(RtArchive, DigestSeparatesTheSignedZeroes) {
  using ddx::rt::Node;
  const std::array<std::string, 1> symbols{"x"};
  const std::array<Node<double>, 1> plus{Node<double>{.value = +0.0}};
  const std::array<Node<double>, 1> minus{Node<double>{.value = -0.0}};

  ASSERT_TRUE(plus[0].value == minus[0].value) << "the two must compare equal";
  EXPECT_NE(ddx::rt::digest<double>(symbols, {}, {}, plus, 1),
            ddx::rt::digest<double>(symbols, {}, {}, minus, 1));
}

// The opcode table is what every opcode byte in the payload *means*, so it is
// inside the checksum: '+' is 0x2B and '/' is 0x2F, one bit apart and the same
// arity, so remapping every Add to a Div passes sound() untouched.
//
// The model deliberately contains no '+' of its own.  Every Add in it is one
// the reverse sweep created, which puts it above model_nodes and so beyond the
// model digest -- the digest covers the model, the checksum covers everything.
// The byte-flip sweep cannot reach this: it needs a specific bit in one label.
TEST(RtArchive, RefusesAFlippedOpcodeLabel) {
  const Scratch file{"label"};
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return sin(x) * cos(x);
  });
  ASSERT_TRUE(eq.save(file).has_value());
  const auto want = *eq.jacobian(0.7);

  auto bytes = file.bytes();
  bool flipped = false;
  for (std::size_t i = ddx::rt::detail::Container::header_bytes;
       i + 9 < bytes.size(); ++i) {
    std::uint64_t len = 0;
    std::memcpy(&len, bytes.data() + i, sizeof len);
    if (len != 1 || static_cast<char>(bytes[i + 8]) != '+') {
      continue;
    }
    bytes[i + 8] ^= std::byte{0x04}; // '+' -> '/'
    flipped = true;
    break;
  }
  ASSERT_TRUE(flipped) << "no one-character \"+\" label in the table";
  file.write(bytes);

  const auto loaded = ddx::rt::load(file.path());
  if (loaded) {
    // Precise, so a reader sees what was at stake.
    EXPECT_EQ(*loaded->jacobian(0.7), want)
        << "an opcode was remapped and the gradient moved";
    FAIL() << "a flipped opcode label was accepted";
  }
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt);
}

// The prologue is the one region a checksum cannot cover, the checksum living
// in it, so every field is verified and none is written that is not.
//
// Swept rather than spot-checked: this catches the *next* field somebody adds
// and forgets.  It found `model_nodes` -- 12 of 168 corruptions accepted.
//
// Exhaustive rather than a few masks: a sampled set is adequate only while
// every field is checked by exact equality, and a range-checked field can
// accept a value some masks never produce.  1.5 s, nearly all of it file I/O.
TEST(RtArchive, EveryPrologueByteIsVerified) {
  const Scratch file{"prologue"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();
  ASSERT_GT(whole.size(), ddx::rt::detail::Container::header_bytes);

  for (std::size_t at = 0; at < ddx::rt::detail::Container::header_bytes;
       ++at) {
    for (int mask = 1; mask < 256; ++mask) {
      auto bytes = whole;
      bytes[at] ^= static_cast<std::byte>(mask);
      file.write(bytes);
      ASSERT_FALSE(ddx::rt::load(file.path()).has_value())
          << "prologue byte " << at << " accepts corruption 0x" << std::hex
          << mask << ": written but not fully verified";
    }
  }
}

// --- the pieces --------------------------------------------------------------

// Opcode bytes are table-order, so the file names them by label and remaps on
// load -- which is what survives appending to DDX_UNARY_MATH_TABLE.
TEST(RtArchive, OpcodesTravelByLabel) {
  const auto labels = ddx::rt::opcode_labels();
  ASSERT_EQ(labels.size(), ddx::rt::op_count);
  for (const auto [i, label] : labels | std::views::enumerate) {
    const auto back = ddx::rt::opcode_of(label);
    ASSERT_TRUE(back.has_value()) << label;
    EXPECT_EQ(static_cast<std::size_t>(*back), static_cast<std::size_t>(i));
  }
  EXPECT_FALSE(ddx::rt::opcode_of("no such op").has_value());
}

// The stamp refuses old files rather than misreading them, so it must actually
// depend on the fields.
TEST(RtArchive, TheSchemaStampSeparatesShapes) {
  constexpr auto snapshot =
      ddx::rt::detail::Container::stamp<ddx::rt::Snapshot<double>>();
  constexpr auto narrower =
      ddx::rt::detail::Container::stamp<ddx::rt::Snapshot<float>>();
  constexpr auto unrelated =
      ddx::rt::detail::Container::stamp<ddx::rt::Coloring>();
  static_assert(snapshot != narrower);
  static_assert(snapshot != unrelated);
  EXPECT_NE(snapshot, 0u);
}

// A saved arena is installed verbatim, not replayed: make() folds and swaps a
// commutative pair, renumbering the ids the saved sweeps name.
TEST(RtArchive, LoadedIdsAreTheSavedIds) {
  const Scratch file{"ids"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  auto snap = ddx::rt::load_snapshot<>(file.path());
  ASSERT_TRUE(snap.has_value()) << snap.error().code;

  // rebuild() consumes the node array, so the comparison keeps its own copy.
  const auto saved = (*snap)->nodes;
  const auto arena = std::move(*snap).rebuild().arena;
  ASSERT_EQ(arena->size(), saved.size());
  for (const auto [i, node] : saved | std::views::enumerate) {
    const auto &back = (*arena)[static_cast<ddx::rt::NodeId>(i)];
    EXPECT_EQ(back.op, node.op) << "node " << i;
    EXPECT_EQ(back.a, node.a) << "node " << i;
    EXPECT_EQ(back.b, node.b) << "node " << i;
    EXPECT_EQ(back.slot, node.slot) << "node " << i;
  }
}

// --- the machine code --------------------------------------------------------
//
// The graph half of a file saves milliseconds; this half saves a compile.  A
// stored kernel runs only where graph, host and options all still agree --
// adopt() cannot tell that an object came from another graph.
#ifdef DDX_HAS_JIT
TEST(RtArchive, CarriesTheCompiledKernel) {
  const Scratch file{"kernel"};
  {
    auto built = coupled();
    // Off by default: a kernel does not keep a megabyte of machine code alive
    // on the chance that someone saves it.
    built.options(
        {.backend = ddx::jit::Backend::Compile, .retain_object = true});
    ASSERT_TRUE(built.wait_for_kernel());
    ASSERT_TRUE(built.save(file).has_value());
  }

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  // No wait_for_kernel(): linked from the file during the freeze, so it is
  // already answering.  That is the whole claim.
  EXPECT_TRUE(loaded->uses_kernel());
  EXPECT_TRUE(loaded->kernel_level().has_value());

  // And it is the same kernel, not merely an equivalent one.
  const auto want = coupled();
  EXPECT_EQ(*loaded->jacobian(kPoint), *want.jacobian(kPoint));
  EXPECT_EQ(*loaded->hessian(kPoint), *want.hessian(kPoint));
}

// `backend` says whether to compile, never what is emitted, so it is policy
// and not part of an Object's identity: an Adapt lane links the kernel a
// Compile run stored rather than counting its way to identical machine code.
TEST(RtArchive, AdaptLinksAKernelStoredUnderCompile) {
  const Scratch file{"adapt_kernel"};
  {
    auto built = coupled();
    built.options(
        {.backend = ddx::jit::Backend::Compile, .retain_object = true});
    ASSERT_TRUE(built.wait_for_kernel());
    ASSERT_TRUE(built.save(file).has_value());
  }

  auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  // A threshold no test could reach, so a kernel here came off the file.
  loaded->options({.backend = ddx::jit::Backend::Adapt,
                   .warm_points = 1uz << 40,
                   .retain_object = true});
  EXPECT_TRUE(loaded->uses_kernel()) << "a stored kernel was made to wait";
  EXPECT_FALSE(loaded->warming().has_value())
      << "still counting toward a rung it already has";

  const auto want = coupled();
  EXPECT_EQ(*loaded->jacobian(kPoint), *want.jacobian(kPoint));
}

TEST(RtArchive, SavesNoKernelWithoutRetainObject) {
  const Scratch file{"no_retain"};
  auto built = coupled();
  built.options(
      {.backend = ddx::jit::Backend::Compile, .retain_object = false});
  ASSERT_TRUE(built.wait_for_kernel());
  ASSERT_TRUE(built.save(file).has_value());

  // A shortfall in what the file is worth rather than a failure.
  const auto snap = ddx::rt::load_snapshot<>(file.path());
  ASSERT_TRUE(snap.has_value());
  EXPECT_TRUE((*snap)->objects.empty());
  EXPECT_TRUE(ddx::rt::load(file.path()).has_value());
}

TEST(RtArchive, RefusesAKernelFromAnotherGraph) {
  const Scratch file{"foreign_kernel"};
  auto built = coupled();
  built.options({.backend = ddx::jit::Backend::Compile, .retain_object = true});
  ASSERT_TRUE(built.wait_for_kernel());

  ASSERT_TRUE(built.save(file).has_value());

  // The right code relabelled as another graph's, written back through the same
  // serialiser so the checksum still holds: a stored kernel run against a graph
  // it was not emitted from is silently wrong arithmetic, and the digest is
  // what gates it.
  auto sound = ddx::rt::load_snapshot<>(file.path());
  ASSERT_TRUE(sound.has_value());
  auto snap = std::move(*sound).release();
  ASSERT_FALSE(snap.objects.empty());
  for (auto &o : snap.objects) {
    o.digest ^= 1;
  }
  ASSERT_TRUE(ddx::rt::save(snap, file.path()).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value());
  // The graph still loads; the code is simply not adopted.
  EXPECT_EQ(*loaded->jacobian(kPoint), *coupled().jacobian(kPoint));
}
#endif

// The pattern is what makes the Jacobian block mean anything, so a file may not
// say something the block cannot mean.  Damage never reaches these scalars --
// the checksum stops it long before the parse -- so each one needs a forged
// case, written through the real serialiser so the CRC agrees with the lie.
//
// The shape is pinned against the roots and the symbols before anything is
// sized by it, which is why `rows + 1` has no wrap case of its own: `rows` is
// an equality against the root count by the time it is incremented.
TEST(RtArchive, RefusesAForgedJacobianPattern) {
  const Scratch file{"forged_pattern"};
  const auto built = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return std::array{x * log(x) + y, y * y + z, exp(x * z)};
  });
  ASSERT_TRUE(built.save(file).has_value());

  {
    const auto clean = ddx::rt::load_snapshot<>(file.path());
    ASSERT_TRUE(clean.has_value()) << clean.error().code;
    ASSERT_LT((*clean)->jacobian.pattern.nonzeros(), 9u)
        << "a dense pattern has no hole to forge";
    ASSERT_EQ((*clean)->jacobian.pattern.begin()[1].size(), 2u)
        << "row 1 needs two columns for the ordering cases";
  }

  // Rewritten from the equation each time: a damaged file left in place would
  // make every later case load something already broken and pass for the wrong
  // reason.
  const auto refuses = [&file, &built](auto damage, std::string_view what) {
    EXPECT_TRUE(built.save(file).has_value()) << what;
    auto sound = ddx::rt::load_snapshot<>(file.path());
    EXPECT_TRUE(sound.has_value()) << what;
    if (!sound) {
      return;
    }
    auto snap = std::move(*sound).release();
    damage(snap.jacobian);
    EXPECT_TRUE(ddx::rt::save(snap, file.path()).has_value()) << what;
    const auto loaded = ddx::rt::load<double, 3>(file.path());
    EXPECT_FALSE(loaded.has_value()) << what;
    if (!loaded) {
      EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt) << what;
    }
  };

  refuses([](ddx::rt::Jacobian &j) { j.pattern.col.back() = 99; },
          "a column outside the symbol table");
  refuses(
      [](ddx::rt::Jacobian &j) {
        // There is no row count to forge -- it is rowptr.size() - 1 -- so the
        // way to claim a row the roots do not have is to carry an offset for
        // it.
        j.pattern.rowptr.push_back(j.pattern.rowptr.back());
      },
      "a row count the roots do not agree with");
  refuses([](ddx::rt::Jacobian &j) { ++j.pattern.rowptr.back(); },
          "a rowptr running past the block");
  refuses([](ddx::rt::Jacobian &j) { j.pattern.rowptr.pop_back(); },
          "one offset short of a row each");
  refuses(
      [](ddx::rt::Jacobian &j) {
        // Row 1's two columns, out of order: at() binary searches, so an
        // unsorted row hides a cell that is really there.
        const auto first = j.pattern.rowptr[1];
        std::swap(j.pattern.col[first], j.pattern.col[first + 1]);
      },
      "a row whose columns descend");
  refuses(
      [](ddx::rt::Jacobian &j) {
        const auto first = j.pattern.rowptr[1];
        j.pattern.col[first + 1] = j.pattern.col[first];
      },
      "a row naming one column twice");
  refuses([](ddx::rt::Jacobian &j) { j.partial.pop_back(); },
          "fewer cells than the pattern names");

  // And the untouched file still loads, or every case above is vacuous.
  {
    ASSERT_TRUE(built.save(file).has_value());
    auto snap = ddx::rt::load_snapshot<>(file.path());
    ASSERT_TRUE(snap.has_value()) << snap.error().code;
    ASSERT_TRUE(ddx::rt::save(**snap, file.path()).has_value());
    const auto loaded = ddx::rt::load<double, 3>(file.path());
    ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
    EXPECT_EQ(*built.jacobian(0.6, 1.4, 0.9), *loaded->jacobian(0.6, 1.4, 0.9));
    EXPECT_EQ(built.jacobian_pattern()->get().nonzeros(),
              loaded->jacobian_pattern()->get().nonzeros());
    EXPECT_EQ(built.jacobian_pattern()->get().col,
              loaded->jacobian_pattern()->get().col);
    EXPECT_EQ(built.jacobian_pattern()->get().rowptr,
              loaded->jacobian_pattern()->get().rowptr);
  }
}

// The digest keys the JIT's object cache, so two graphs it cannot tell apart
// would be handed each other's machine code.  A select's third operand is the
// one that folding `a` and `b` alone would have missed.
//
// Every operand is a leaf and the root mentions both candidate arms, so the two
// graphs have identical node arrays, identical ids, identical liveness and
// identical contraction -- the false arm is the only difference there is.
// Getting there took three attempts, and the two that did not bite are worth
// recording: arms that are *different expressions* separate the digest through
// their own nodes, and a `.build_jacobian()` separates it through the
// derivative nodes, since the sweep reaches a different arm in each.  A
// multiply as an arm separates it a third way, by changing which node gets
// swallowed into an fma and so changing contracted_order().
TEST(RtArchive, DigestSeparatesASelectsFalseArm) {
  const auto digest_of = [](bool other_arm) {
    ddx::rt::Builder<double> b;
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    const auto p = var(b, "p");
    const auto q = var(b, "q");
    const auto chosen = select(lt(x, y), p, other_arm ? q : p);
    return ddx::rt::digest(
        ddx::rt::GraphBuilder<double>{b}.value(chosen + p + q).finish());
  };
  EXPECT_NE(digest_of(false), digest_of(true))
      << "two selects differing only in the false arm shared a cache key";
}
