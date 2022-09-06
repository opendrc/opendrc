#include <odrc/interface/gdsii/gdsii.hpp>

#include <cstddef>
#include <exception>
#include <fstream>
#include <vector>

namespace odrc {
void gdsii_library::read(const std::filesystem::path& file_path) {
  std::vector<std::byte> buffer(65536);
  std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
  if (not ifs) {
    throw std::runtime_error("Cannot open " + file_path.string() + ": " +
                             std::strerror(errno));
  }
  while (true) {
    // read record header
    ifs.read(reinterpret_cast<char*>(buffer.data()), 4);
    int record_length = _parse_gdsii_int16(&buffer[0]);
    gdsii_record_type record_type = static_cast<gdsii_record_type>(buffer[2]);
    int data_type = std::to_integer<int>(buffer[3]);
    ifs.read(reinterpret_cast<char*>(buffer.data() + 4), record_length - 4);

    switch (record_type) {
      case gdsii_record_type::HEADER: {
        gdsii_version = _parse_gdsii_int16(&buffer[4]);
        break;
      }
      default:
        break;
    }
    break;  // for now, just to demonstrate how to parse a record
            // head
  }
}
int gdsii_library::_parse_gdsii_int16(std::byte* p) {
  return (std::to_integer<int>(p[0]) << 8) | std::to_integer<int>(p[1]);
}
}  // namespace odrc