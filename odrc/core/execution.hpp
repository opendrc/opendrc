#pragma once

namespace odrc::execution {
class sequenced_policy {};
class parallel_policy {};

constexpr sequenced_policy seq;
constexpr parallel_policy  par;
}  // namespace odrc::execution