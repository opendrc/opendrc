#pragma once

#include <odrc/gdsii/gdsii.hpp>

namespace odrc {
    void width_check(const odrc::gdsii::library &lib, int layer, int threshold);
}