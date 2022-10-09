#pragma once

#include <odrc/interface/gdsii/gdsii.hpp>

namespace odrc {
    void width_check(const odrc::gdsii::library &lib, int layer, int threshold);
}