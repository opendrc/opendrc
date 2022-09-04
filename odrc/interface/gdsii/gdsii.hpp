#pragma once

#include <cstddef>
#include <filesystem>
#include <type_traits>

namespace odrc
{
  enum class gdsii_record_type : std::underlying_type_t<std::byte>
  {
    HEADER = 0x00,
  };
  enum class gdsii_data_type : std::underlying_type_t<std::byte>
  {
    two_byte_signed_integer = 0x02,
    four_byte_signed_integer = 0x03,
  };

  class gdsii_library
  {
  public:
    void read(const std::filesystem::path &file_path);
    int gdsii_version;

  private:
    int _parse_gdsii_int16(std::byte *p);
  };
}